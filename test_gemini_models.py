import os
import time
from pathlib import Path

from google import genai
from dotenv import load_dotenv


# ---------------------------------------------------------
# Find the project root regardless of the current terminal
# directory.
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / "backend" / ".env"

load_dotenv(ENV_FILE)


api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        f"GEMINI_API_KEY not found. Checked: {ENV_FILE}"
    )

print(f"Loaded API key from: {ENV_FILE}")

client = genai.Client(api_key=api_key)


models = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-2.5-flash",
]


for model_name in models:
    print("\n" + "=" * 60)
    print(f"Testing: {model_name}")
    print("=" * 60)

    start = time.perf_counter()

    try:
        response = client.models.generate_content(
            model=model_name,
            contents="Reply with exactly: OK",
        )

        elapsed = time.perf_counter() - start

        print(f"SUCCESS: {elapsed:.2f}s")
        print(f"Response: {response.text}")

    except Exception as e:
        elapsed = time.perf_counter() - start

        print(f"FAILED: {elapsed:.2f}s")
        print(f"{type(e).__name__}: {e}")