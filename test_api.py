import requests

BASE_URL = "http://127.0.0.1:5000"

# Test 1 - health check
print("Testing health endpoint...")
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Test 2 - full report
print("\nTesting report endpoint...")
with open("./notebooks/test_upload.csv", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/report",
        files={"file": ("test_upload.csv", f, "text/csv")}
    )

if response.status_code == 200:
    with open("test_report.pdf", "wb") as f:
        f.write(response.content)
    print("Report saved to test_report.pdf")
    print("Open it and check it looks correct")
else:
    print("Error:", response.status_code)
    print(response.json())