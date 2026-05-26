from google import genai
import os

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

for model in ["gemini-1.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-flash"]:
    print(f"Testing {model}...")
    try:
        response = client.models.generate_content(
            model=model,
            contents="Say hello in one word"
        )
        print(f"SUCCESS: {response.text.strip()}")
    except Exception as e:
        print(f"FAILED: {e}")
