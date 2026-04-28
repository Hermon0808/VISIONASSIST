from google import genai

client = genai.Client(api_key="AIzaSyBklb6ht0gRnosWb-0QbQi2seAqUZoYaw0")

try:
    response = client.models.generate_content(
        model="gemini-2.0-flash",  # ← change is HERE
        contents="Say API is working"
    )

    print("✅ API is working")
    print(response.text)

except Exception as e:
    print("❌ API is NOT working")
    print("Error:", str(e))