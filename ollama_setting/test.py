from ollama import Client

PORT = 50001
MODEL = "gpt-oss:20b"

client = Client(host=f"http://localhost:{PORT}")

prompts = [
    "Explain transformers briefly.",
    "What is reinforcement learning?",
    "Describe gradient descent."
]

for p in prompts:

    response = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": p}],
        think="low",
        options={
            "temperature": 0,
            "num_predict": 256
        }
    )

    print("PROMPT:", p)
    print("REASONING:", response["message"]["thinking"])
    print("ANSWER:", response["message"]["content"])
    print("-" * 50)