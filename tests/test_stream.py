import httpx

url = "http://127.0.0.1:8000/query/stream"
payload = {"question": "Wie funktioniert eine List Comprehension?"}

# timeout=None verhindert, dass httpx während des Streamings abbricht
with httpx.Client(timeout=None) as client:
    with client.stream("POST", url, json=payload) as response:
        for chunk in response.iter_text():
            print(chunk, end="", flush=True)
print()