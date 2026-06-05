from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api = os.getenv("groq_api_key")
client = OpenAI(api_key=api, base_url="https://api.groq.com/openai/v1")

history = [{'role': 'system', 'content': 'you are a helpful assistant'}]

while True:
    user = input("You: ")
    if user.lower() == "exit":
        break

    history.append({'role': 'user', 'content': user})

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=history
        )
        reply = response.choices[0].message.content
        print(f"Assistant: {reply}")
        history.append({'role': 'assistant', 'content': reply})

        if response.usage.total_tokens > 1000:
            print("[summarising history...]")
            recent = history[-6:] if len(history) > 6 else history[1:]
            compact = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": f"summarise this briefly keeping all important info: {recent}"}]
            )
            history = [history[0], {'role': 'assistant', 'content': compact.choices[0].message.content}]

    except Exception as e:
        print(f"error: {e}")
        history.pop()