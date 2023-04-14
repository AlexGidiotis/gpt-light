import requests

payload = {
    "prompt_txt": "\n"
}

r = requests.get("http://127.0.0.1:8000/")
print(r.text)
r = requests.post("http://127.0.0.1:8000/lightchatgpt/", json=payload)
print(r.text)
