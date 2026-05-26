from google import genai
import os

keys = [
    os.environ.get("GOOGLE_API_KEY"),
    os.environ.get("GOOGLE_API_KEY_2"),
    os.environ.get("GOOGLE_API_KEY_3")
]

for i, key in enumerate(keys, 1):
    print(f"Testing Key {i}...")
    if not key:
        print(f"Key {i} is missing from environment!")
        continue
    try:
        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Say hello in one word"
        )
        print(f"Key {i} SUCCESS: {response.text.strip()}")
    except Exception as e:
        print(f"Key {i} FAILED: {e}")
