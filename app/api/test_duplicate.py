import requests
import hashlib

API_URL = "http://localhost:8000"
FILE_PATH = "/home/zenteiq/AI-QUIZ-BACKEND/uploads/20251104_061229_Machine_Learning_Article.pdf"  # Use your actual PDF

def calculate_local_hash(filepath):
    """Calculate hash locally to compare"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def upload_file(filepath):
    """Upload file and return response"""
    with open(filepath, "rb") as f:
        files = {"file": f}
        response = requests.post(f"{API_URL}/api/upload", files=files)
    return response.json()

# Test
print("📤 Uploading file first time...")
response1 = upload_file(FILE_PATH)
print(f"Response 1: ID={response1['id']}, filename={response1['filename']}")

print("\n📤 Uploading same file again...")
response2 = upload_file(FILE_PATH)
print(f"Response 2: ID={response2['id']}, filename={response2['filename']}")

print("\n🔍 Comparison:")
print(f"IDs match: {response1['id'] == response2['id']}")
print(f"Timestamps match: {response1['uploaded_at'] == response2['uploaded_at']}")

if response1['id'] == response2['id']:
    print("✅ Duplicate detection is WORKING!")
else:
    print("❌ Duplicate detection is NOT working!")
    print(f"\nLocal file hash: {calculate_local_hash(FILE_PATH)}")