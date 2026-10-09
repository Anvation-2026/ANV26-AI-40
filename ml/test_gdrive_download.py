import requests
import re
import sys

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
})

url = "https://drive.google.com/uc?export=download&id=1r-oNYTPiPCOUzSjChjCIYTdkjBTugqxR"
r1 = session.get(url)
print("r1 status:", r1.status_code)

uuid_match = re.search(r'name=["\']uuid["\']\s+value=["\']([^"\']+)["\']', r1.text)
confirm_match = re.search(r'name=["\']confirm["\']\s+value=["\']([^"\']+)["\']', r1.text)

uuid_val = uuid_match.group(1) if uuid_match else None
confirm_val = confirm_match.group(1) if confirm_match else "t"

print("uuid:", uuid_val)
print("confirm:", confirm_val)

url2 = f"https://drive.usercontent.google.com/download?id=1r-oNYTPiPCOUzSjChjCIYTdkjBTugqxR&export=download&confirm={confirm_val}"
if uuid_val:
    url2 += f"&uuid={uuid_val}"

r2 = session.get(url2, stream=True)
print("r2 status:", r2.status_code)
for k, v in r2.headers.items():
    if k.lower() in ["content-type", "content-length", "content-disposition"]:
        print(f"{k}: {v}")
chunk = next(r2.iter_content(chunk_size=1024), None)
print("First chunk size:", len(chunk) if chunk else 0)
