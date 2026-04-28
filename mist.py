from mistralai import Mistral

api_key = "ASDetPwVpMMHtcWbvGnAnZMccNezr7fz"

client = Mistral(api_key=api_key)

try:
    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {"role": "user", "content": "Say API is working"}
        ]
    )

    print("✅ API is working")
    print(response.choices[0].message.content)

except Exception as e:
    print("❌ API is NOT working")
    print("Error:", str(e))