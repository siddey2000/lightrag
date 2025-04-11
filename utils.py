import ollama
from PyPDF2 import PdfReader
import numpy as np
from config import OLLAMA_EMBEDDING_MODEL, PDF_FILE_PATH, CHUNK_SIZE, CHUNK_OVERLAP, OLLAMA_LLM_MODEL  # Import the new variable
from typing import Union, AsyncIterator, Dict  # Import Union

async def _ollama_model_if_cache(  # Renamed function
    model_name: str,
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[Dict[str, str]] | None = None,
    **kwargs: Dict,  # More specific type hint for kwargs
) -> Union[str, AsyncIterator[str]]:
    """
    Handles communication with the Ollama LLM.

    Args:
        model_name: The name of the Ollama model to use.
        prompt: The user's input query.
        system_prompt: Optional system prompt to provide context.
        history_messages: Optional list of previous messages in the conversation.
        **kwargs: Additional keyword arguments to pass to the Ollama API.

    Returns:
        The LLM's response, either as a single string or an asynchronous stream of strings.
    """
    stream = True if kwargs.get("stream") else False

    kwargs.pop("max_tokens", None)
    # kwargs.pop("response_format", None) # allow json
    host = kwargs.pop("host", None)
    timeout = kwargs.pop("timeout", None)
    kwargs.pop("hashing_kv", None)
    api_key = kwargs.pop("api_key", None)
    headers = {
        "Content-Type": "application/json",
        "User-Agent": f"LightRAG/0.1.0",  # Replace with your actual version
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    ollama_client = ollama.AsyncClient(host=host, timeout=timeout, headers=headers)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    response = await ollama_client.chat(model=model_name, messages=messages, **kwargs)
    if stream:
        """cannot cache stream response and process reasoning"""

        async def inner():
            async for chunk in response:
                yield chunk["message"]["content"]

        return inner()
    else:
        model_response = response["message"]["content"]

        """
        If the model also wraps its thoughts in a specific tag,
        this information is not needed for the final
        response and can simply be trimmed.
        """

        return model_response


async def ollama_model_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[Dict[str, str]] | None = None,
    keyword_extraction: bool = False,
    **kwargs: Dict,
) -> Union[str, AsyncIterator[str]]:
    """
    Generates a completion (response) from the Ollama language model.

    Args:
        prompt: The user's input query.
        system_prompt: Optional system prompt to provide context to the LLM.
        history_messages: Optional list of previous messages in the conversation.
        keyword_extraction: If True, requests the LLM to provide the response in JSON format.
        **kwargs: Additional keyword arguments to pass to the Ollama API.

    Returns:
        The LLM's response, either as a single string or an asynchronous stream of strings.
    """
    if keyword_extraction:
        kwargs["format"] = "json"
    model_name = OLLAMA_LLM_MODEL  # Use the config variable
    return await _ollama_model_if_cache(  # Use the renamed function
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

def read_pdf(file_path=PDF_FILE_PATH):  # Use the config variable
    text = ""
    try:
        with open(file_path, 'rb') as file:
            print("File opened successfully (inside read_pdf)")  # Debug print
            reader = PdfReader(file)
            print(f"PdfReader created: {reader}")  # Debug print
            for page_num in range(min(8, len(reader.pages))):
                print(f"Processing page: {page_num}")  # Debug print
                page = reader.pages[page_num]
                text_from_page = page.extract_text()
                if text_from_page:
                    text += text_from_page
                else:
                    print(f"No text extracted from page: {page_num}")  # Debug print
    except FileNotFoundError as e:
            print(f"FileNotFoundError: {e}")  # Print the specific error
            return None
    except Exception as e:
            print(f"Error opening file: {e}")  # Catch other potential errors
            return None
    return text


def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):  # Use config variables
    chunks = []
    chunk_ids = []
    start = 0
    chunk_num = 1
    while start < len(text):
        end = start + chunk_size
        if end > len(text):
            end = len(text)
        chunk = text[start:end]
        chunks.append(chunk)
        chunk_ids.append(chunk_num)
        if start >= len(text):  # Combined if conditions
            break
        start = max(0, end - chunk_overlap)
        chunk_num += 1
    return chunks, chunk_ids


def generate_embeddings(texts, model_name=OLLAMA_EMBEDDING_MODEL):  # Use config variable
    client = ollama.Client()
    embeddings = []
    for text in texts:
        try:
            response = client.embeddings(model=model_name, prompt=text)
            embeddings.append(response['embedding'])
        except Exception as e:
            print(f"Error generating embedding for text: {text[:50]}...: {e}")
            embeddings.append(None)  # Placeholder for failed embedding
    return np.array(embeddings)


