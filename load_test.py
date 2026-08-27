"""Load Test -- 2,000-request benchmark for the semantic cache.

Runs entirely in-process (no server or Ollama needed). Seeds the cache
with varied prompts, then hammers it with a realistic mix of similar,
repeated, and unique queries. Outputs a portfolio-ready report.

Usage:
    python load_test.py
"""

import math
import random
import time

from src.semantic_cache.config import CacheConfig
from src.semantic_cache.cache_engine import SemanticCache


# ── Test Prompts ────────────────────────────────────────────────────
# 50 seed prompts covering different categories
SEED_PROMPTS = [
    # Science & Math
    ("What is photosynthesis?", "Photosynthesis is the process by which plants convert sunlight, water, and CO2 into glucose and oxygen."),
    ("Explain the theory of relativity", "Einstein's theory of relativity describes how space and time are linked for objects moving at consistent speeds."),
    ("What is the Pythagorean theorem?", "The Pythagorean theorem states that in a right triangle, a^2 + b^2 = c^2."),
    ("How does gravity work?", "Gravity is a fundamental force that attracts objects with mass toward each other."),
    ("What is DNA?", "DNA (deoxyribonucleic acid) is a molecule that carries genetic instructions for life."),
    ("Explain quantum mechanics", "Quantum mechanics describes the behavior of matter and energy at the atomic and subatomic level."),
    ("What is the speed of light?", "The speed of light in vacuum is approximately 299,792,458 meters per second."),
    ("How do black holes form?", "Black holes form when massive stars collapse under their own gravity at the end of their life cycle."),
    ("What is evolution?", "Evolution is the change in inherited characteristics of biological populations over successive generations."),
    ("Explain the water cycle", "The water cycle describes how water evaporates, forms clouds, precipitates, and flows back to water bodies."),

    # Programming
    ("What is Python?", "Python is a high-level, interpreted programming language known for its simple syntax and versatility."),
    ("Explain what an API is", "An API (Application Programming Interface) is a set of protocols for building and integrating software."),
    ("What is machine learning?", "Machine learning is a subset of AI where systems learn from data to improve performance without explicit programming."),
    ("How does a database work?", "A database stores organized data that can be accessed, managed, and updated electronically."),
    ("What is version control?", "Version control tracks changes to files over time, allowing multiple people to collaborate on code."),
    ("Explain REST APIs", "REST is an architectural style for designing networked applications using standard HTTP methods."),
    ("What is a neural network?", "A neural network is a computing system inspired by biological neural networks in the brain."),
    ("How does encryption work?", "Encryption converts readable data into coded form using algorithms, protecting it from unauthorized access."),
    ("What is cloud computing?", "Cloud computing delivers computing services like servers, storage, and databases over the internet."),
    ("Explain Docker containers", "Docker containers package applications with their dependencies into standardized units for deployment."),

    # History & Geography
    ("Who was Albert Einstein?", "Albert Einstein was a theoretical physicist who developed the theory of relativity."),
    ("What caused World War I?", "WWI was triggered by the assassination of Archduke Franz Ferdinand and fueled by alliances and imperialism."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("Explain the Industrial Revolution", "The Industrial Revolution was the transition from agricultural to manufacturing economies in the 18th-19th centuries."),
    ("Who discovered America?", "Christopher Columbus reached the Americas in 1492, though indigenous peoples had lived there for millennia."),

    # General Knowledge
    ("What is climate change?", "Climate change refers to long-term shifts in global temperatures and weather patterns."),
    ("How does the internet work?", "The internet is a global network of computers that communicate using standardized protocols like TCP/IP."),
    ("What is cryptocurrency?", "Cryptocurrency is a digital currency that uses cryptography for security and operates on blockchain technology."),
    ("Explain how vaccines work", "Vaccines train the immune system to recognize and fight specific pathogens by introducing harmless components."),
    ("What is artificial intelligence?", "Artificial intelligence is the simulation of human intelligence by computer systems."),

    # Business & Economics
    ("What is inflation?", "Inflation is the rate at which the general price level of goods and services rises over time."),
    ("Explain supply and demand", "Supply and demand is an economic model where price is determined by the relationship between availability and desire."),
    ("What is GDP?", "GDP (Gross Domestic Product) measures the total value of goods and services produced in a country."),
    ("How does the stock market work?", "The stock market is where shares of public companies are traded between buyers and sellers."),
    ("What is blockchain?", "Blockchain is a distributed ledger technology that records transactions across many computers securely."),

    # More technical
    ("What is a hash function?", "A hash function maps data of arbitrary size to fixed-size values, used in data integrity and cryptography."),
    ("Explain TCP vs UDP", "TCP provides reliable ordered delivery; UDP provides faster but unreliable connectionless communication."),
    ("What is a load balancer?", "A load balancer distributes network traffic across multiple servers to ensure reliability and performance."),
    ("How does caching work?", "Caching stores copies of frequently accessed data in fast storage to reduce latency and server load."),
    ("What is microservices architecture?", "Microservices breaks applications into small independent services that communicate via APIs."),

    # Simple questions
    ("What is 2 + 2?", "2 + 2 equals 4."),
    ("What color is the sky?", "The sky appears blue due to Rayleigh scattering of sunlight in the atmosphere."),
    ("How many continents are there?", "There are 7 continents: Africa, Antarctica, Asia, Australia, Europe, North America, and South America."),
    ("What is the largest ocean?", "The Pacific Ocean is the largest ocean, covering about 63 million square miles."),
    ("How many planets are in our solar system?", "There are 8 planets in our solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune."),

    # More varied
    ("What is a compiler?", "A compiler translates source code written in a programming language into machine code."),
    ("Explain the difference between HTTP and HTTPS", "HTTPS is HTTP with encryption via TLS/SSL, providing secure data transfer between client and server."),
    ("What is an operating system?", "An operating system manages computer hardware and software resources and provides services to programs."),
    ("How does Wi-Fi work?", "Wi-Fi uses radio waves to provide wireless high-speed internet and network connections."),
    ("What is a firewall?", "A firewall monitors and controls incoming and outgoing network traffic based on security rules."),
]

# Similar variations for cache-hit testing
SIMILAR_VARIANTS = [
    # Variations of seed prompts (semantically similar but different words)
    "Explain photosynthesis to me",
    "Can you describe the theory of relativity?",
    "Tell me about the Pythagorean theorem",
    "How does gravitational force work?",
    "What is DNA and what does it do?",
    "Describe quantum mechanics in simple terms",
    "How fast does light travel?",
    "How are black holes created?",
    "What is the theory of evolution?",
    "Describe the water cycle process",
    "Tell me about Python programming language",
    "What exactly is an API?",
    "Explain machine learning to me",
    "How do databases function?",
    "What is git version control?",
    "Explain RESTful APIs",
    "What are neural networks?",
    "How does data encryption work?",
    "What is cloud computing technology?",
    "Explain what Docker containers are",
    "Tell me about Albert Einstein",
    "What were the causes of World War 1?",
    "What is Tokyo the capital of?",
    "Describe the Industrial Revolution",
    "Who discovered the Americas?",
    "What is global climate change?",
    "How does the internet function?",
    "What is crypto currency?",
    "How do vaccines protect us?",
    "What is AI?",
    "What causes inflation?",
    "How does supply and demand work?",
    "What does GDP measure?",
    "How does the stock exchange work?",
    "What is blockchain technology?",
    "What is a cryptographic hash function?",
    "What is the difference between TCP and UDP?",
    "What does a load balancer do?",
    "How does data caching work?",
    "What is a microservices architecture?",
    "What is two plus two?",
    "Why is the sky blue?",
    "How many continents exist?",
    "Which is the biggest ocean?",
    "How many planets orbit the sun?",
    "What is a code compiler?",
    "How is HTTPS different from HTTP?",
    "What is an OS?",
    "How does wireless internet work?",
    "What is a network firewall?",
]

# Completely unique prompts (guaranteed cache misses)
UNIQUE_PROMPTS = [
    "What is the melting point of tungsten?",
    "Describe the political system of ancient Rome",
    "How do noise-canceling headphones work?",
    "What is the Fibonacci sequence used for?",
    "Explain the concept of opportunity cost",
    "How do solar panels generate electricity?",
    "What is the difference between mitosis and meiosis?",
    "Explain the Doppler effect",
    "What is a Turing machine?",
    "How do airplanes generate lift?",
]


def run_load_test(total_requests: int = 2000) -> dict:
    """Run the load test and return results."""

    print()
    print("=" * 60)
    print("  SEMANTIC CACHE LOAD TEST")
    print("=" * 60)
    print()
    print("  Initializing cache engine...")
    print("  (Loading embedding model -- first time takes ~15 seconds)")
    print()

    # Setup
    config = CacheConfig(
        similarity_threshold=0.85,
        enable_adaptive_threshold=False,
    )
    cache = SemanticCache(config)

    # ── Phase A: Seed the cache ────────────────────────────────────
    print(f"  Seeding cache with {len(SEED_PROMPTS)} entries...")
    seed_start = time.perf_counter()

    for prompt, response in SEED_PROMPTS:
        cache.store(
            prompt=prompt,
            response={"content": response},
            model="gpt-4o",
            model_id="gpt-4o",
            token_usage={
                "prompt_tokens": max(1, len(prompt) // 4),
                "completion_tokens": max(1, len(response) // 4),
            },
            ttl_seconds=86400,
        )

    seed_time = time.perf_counter() - seed_start
    print(f"  Seeded in {seed_time:.1f}s")
    print()

    # ── Phase B: Build request mix ─────────────────────────────────
    # 70% similar variants (should be cache hits)
    # 20% exact repeats (should be cache hits)
    # 10% unique (should be cache misses)
    requests = []

    num_similar = int(total_requests * 0.70)
    num_repeat = int(total_requests * 0.20)
    num_unique = total_requests - num_similar - num_repeat

    for _ in range(num_similar):
        requests.append(("similar", random.choice(SIMILAR_VARIANTS)))

    for _ in range(num_repeat):
        prompt, _ = random.choice(SEED_PROMPTS)
        requests.append(("repeat", prompt))

    for _ in range(num_unique):
        requests.append(("unique", random.choice(UNIQUE_PROMPTS)))

    random.shuffle(requests)

    # ── Phase C: Execute requests ──────────────────────────────────
    print(f"  Running {total_requests:,} requests...")
    print(f"    Similar variants: {num_similar:,} (expect hits)")
    print(f"    Exact repeats:    {num_repeat:,} (expect hits)")
    print(f"    Unique prompts:   {num_unique:,} (expect misses)")
    print()

    hit_latencies = []
    miss_latencies = []
    hits = 0
    misses = 0
    hit_scores = []

    test_start = time.perf_counter()

    for i, (req_type, prompt) in enumerate(requests):
        t0 = time.perf_counter()
        result = cache.lookup(prompt=prompt, model="gpt-4o")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if result.hit:
            hits += 1
            hit_latencies.append(elapsed_ms)
            hit_scores.append(result.similarity_score)
        else:
            misses += 1
            miss_latencies.append(elapsed_ms)

        # Progress indicator every 500 requests
        if (i + 1) % 500 == 0:
            current_rate = hits / (i + 1) * 100
            print(f"    [{i+1:,}/{total_requests:,}] Hit rate: {current_rate:.1f}%")

    test_time = time.perf_counter() - test_start
    total_time = time.perf_counter() - seed_start

    # ── Phase D: Calculate results ─────────────────────────────────
    hit_rate = hits / total_requests
    hit_lat_sorted = sorted(hit_latencies)
    miss_lat_sorted = sorted(miss_latencies)

    def pct(vals, p):
        if not vals:
            return 0.0
        k = (len(vals) - 1) * (p / 100)
        f, c = math.floor(k), math.ceil(k)
        if f == c:
            return vals[int(k)]
        return vals[f] * (c - k) + vals[c] * (k - f)

    avg_hit = sum(hit_latencies) / len(hit_latencies) if hit_latencies else 0.0
    avg_miss = sum(miss_latencies) / len(miss_latencies) if miss_latencies else 0.0
    avg_score = sum(hit_scores) / len(hit_scores) if hit_scores else 0.0

    # Cost projections (GPT-4o pricing)
    avg_prompt_tokens = 25  # typical short prompt
    avg_completion_tokens = 75  # typical response
    cost_per_request = (avg_prompt_tokens * 0.0025 + avg_completion_tokens * 0.010) / 1000
    cost_without_cache = cost_per_request * 1000  # per 1K requests
    cost_with_cache = cost_per_request * 1000 * (1 - hit_rate)
    monthly_requests = 100_000 * 30  # 100K/day

    results = {
        "total_requests": total_requests,
        "hits": hits,
        "misses": misses,
        "hit_rate": hit_rate,
        "test_duration_s": test_time,
        "total_duration_s": total_time,
        "hit_p50": pct(hit_lat_sorted, 50),
        "hit_p95": pct(hit_lat_sorted, 95),
        "hit_p99": pct(hit_lat_sorted, 99),
        "miss_p50": pct(miss_lat_sorted, 50),
        "miss_p95": pct(miss_lat_sorted, 95),
        "miss_p99": pct(miss_lat_sorted, 99),
        "avg_hit_ms": avg_hit,
        "avg_miss_ms": avg_miss,
        "avg_hit_score": avg_score,
        "cost_per_1k_without": cost_without_cache,
        "cost_per_1k_with": cost_with_cache,
        "monthly_without": cost_per_request * monthly_requests,
        "monthly_with": cost_per_request * monthly_requests * (1 - hit_rate),
    }

    # ── Phase E: Print report ──────────────────────────────────────
    print()
    print("=" * 60)
    print("  LOAD TEST REPORT")
    print("=" * 60)
    print()
    print(f"  Test Duration:     {test_time:.1f}s ({total_requests/test_time:.0f} req/s)")
    print(f"  Total Requests:    {total_requests:,}")
    print(f"  Seed Entries:      {len(SEED_PROMPTS)}")
    print()
    print("  CACHE PERFORMANCE")
    print("  " + "-" * 40)
    print(f"  Cache Hits:        {hits:,} ({hit_rate*100:.1f}%)")
    print(f"  Cache Misses:      {misses:,} ({(1-hit_rate)*100:.1f}%)")
    print(f"  Avg Hit Score:     {avg_score:.4f}")
    print()
    print("  LATENCY")
    print("  " + "-" * 40)
    print(f"  Hit  P50:  {pct(hit_lat_sorted, 50):>7.2f}ms")
    print(f"  Hit  P95:  {pct(hit_lat_sorted, 95):>7.2f}ms")
    print(f"  Hit  P99:  {pct(hit_lat_sorted, 99):>7.2f}ms")
    print(f"  Miss P50:  {pct(miss_lat_sorted, 50):>7.2f}ms")
    print(f"  Miss P95:  {pct(miss_lat_sorted, 95):>7.2f}ms")
    print(f"  Miss P99:  {pct(miss_lat_sorted, 99):>7.2f}ms")
    print()
    print("  COST SAVINGS (GPT-4o pricing)")
    print("  " + "-" * 40)
    print(f"  Per 1K requests (no cache):   ${cost_without_cache:.4f}")
    print(f"  Per 1K requests (with cache): ${cost_with_cache:.4f}")
    print(f"  Savings per 1K requests:      ${cost_without_cache - cost_with_cache:.4f} ({hit_rate*100:.1f}%)")
    print()
    print(f"  PROJECTED MONTHLY SAVINGS (100K requests/day)")
    print("  " + "-" * 40)
    print(f"  Without cache:  ${results['monthly_without']:>10.2f}/month")
    print(f"  With cache:     ${results['monthly_with']:>10.2f}/month")
    print(f"  SAVINGS:        ${results['monthly_without'] - results['monthly_with']:>10.2f}/month ({hit_rate*100:.1f}%)")
    print()
    print("=" * 60)
    print()

    return results


if __name__ == "__main__":
    run_load_test(2000)