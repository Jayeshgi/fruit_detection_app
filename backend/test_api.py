"""Quick test script for the /predict endpoint."""
import urllib.request
import json
import os

# Build multipart form data manually
img_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "model", "data", "fruits-360", "Test", "Apple 10", "r0_103_100.jpg"
)

with open(img_path, "rb") as f:
    img_data = f.read()

boundary = "----TestBoundary12345"
body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="apple.jpg"\r\n'
    f"Content-Type: image/jpeg\r\n"
    f"\r\n"
).encode() + img_data + f"\r\n--{boundary}--\r\n".encode()

req = urllib.request.Request(
    "http://127.0.0.1:8000/predict",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
)

resp = urllib.request.urlopen(req)
data = json.loads(resp.read().decode())
print(json.dumps(data, indent=2))
