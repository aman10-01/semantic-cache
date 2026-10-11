import httpx
import time

BASE = "http://localhost:8000"

seeds = [
    ("What is Python?", "Python is a high-level programming language known for its simple syntax."),
    ("Explain quantum computing", "Quantum computing uses quantum bits to perform calculations exponentially faster."),
    ("How does photosynthesis work?", "Plants convert sunlight, water and CO2 into glucose and oxygen."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("Who was Albert Einstein?", "Einstein was a theoretical physicist who developed the theory of relativity."),
]

queries = [
    "Explain the Python programming language",
    "Can you explain quantum computing simply?",
    "How do plants do photosynthesis?",
    "Tell me the capital of Japan",
    "What did Albert Einstein do?",
]

with httpx.Client(timeout=30.0) as c:
    print("Seeding cache...")
    for prompt, resp in seeds:
        r = c.post(f"{BASE}/v1/cache/seed", json={"prompt": prompt, "response": resp})
        print(f"  Seeded: {prompt}")

    print("\nSending similar queries (should be cache hits)...")
    for q in queries:
        r = c.post(f"{BASE}/v1/chat/completions", json={"model": "gpt-4o", "messages": [{"role": "user", "content": q}]})
        status = r.headers.get("X-Cache", "UNKNOWN")
        print(f"  '{q}' -> {status}")
        time.sleep(0.3)

print("\nDone! Check http://localhost:8000/v1/dashboard")