import ollama
from config import OLLAMA_LLM_MODEL

while True:
    user_query = input("You: ")
    if user_query.lower() == 'quit':
        break

    response = ollama.chat(
        model=OLLAMA_LLM_MODEL,  # Make sure this is a model you have in Ollama
        messages=[
            {
                'role': 'user',
                'content': user_query,
            },
        ]
    )

    print(f"Ollama: {response['message']['content']}")

print("Chat ended.")