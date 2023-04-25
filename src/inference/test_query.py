import requests
import json

payload = {
    "prompt_txt": """
    Elaborate on the sentence.\n
    Here's the sentence: The latest advances in AI (GPT, LLM, transformers, etc.) are like a Nokia phone in the 90's.\n
    """
}

r = requests.get("http://127.0.0.1:8000/")
print(r.text)
r = requests.post("http://127.0.0.1:8000/lightchatgpt/", json=payload)
response = json.loads(r.text)
print(response["prompt"])
print(response["out"])
