import httpx
import time

URL = "http://localhost:8000/v1/chat/completions"

prompts = [
    "What is Python?",
    "Explain quantum computing",
    "How does photosynthesis work?",
    "What is the capital of Japan?",
    "Who was Albert Einstein?",
    "Explain the Python programming language",
    "Can you explain quantum computing simply?",
    "How do plants do photosynthesis?",
    "Tell me the capital of Japan",
    "What did Albert Einstein do?",
]

print("Sending requests through proxy at http://localhost:8000...")
with httpx.Client(timeout=30.0) as client:
    for i, prompt in enumerate(prompts, 1):
        try:
            resp = client.post(
                URL,
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            status = resp.headers.get("X-Cache", "UNKNOWN")
            print(f"[{i}/{len(prompts)}] '{prompt}' -> {status}")
            time.sleep(0.5)
        except Exception as e:
            print(f"Error connecting to proxy: {e}")
            break

print("\nDone! Check your browser dashboard now.")