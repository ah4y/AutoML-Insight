"""Test Jupyter Server Connection."""

from utils.jupyter_client import JupyterServerClient

# Your Jupyter server details
SERVER_URL = "http://localhost:8888"
TOKEN = "2b5435e3ecc3a194f7d7b59ac02a14c957c2f0892c9b37fe"

print("=" * 60)
print("Testing Jupyter Server Connection")
print("=" * 60)

# Create client
print(f"\n1. Creating client for {SERVER_URL}...")
client = JupyterServerClient(SERVER_URL, TOKEN)

# Test connection
print("\n2. Testing connection...")
if client.test_connection():
    print("   ✅ Connection successful!")
else:
    print("   ❌ Connection failed!")
    exit(1)

# Get server info
print("\n3. Getting server info...")
info = client.get_server_info()
print(f"   Server version: {info.get('version', 'Unknown')}")

# Test simple code execution
print("\n4. Testing code execution...")
test_code = """
print("Hello from Jupyter!")
result = 2 + 2
print(f"2 + 2 = {result}")
"""

result = client.execute_code_via_file(test_code, timeout=30)

if result.get('success'):
    print("   ✅ Execution successful!")
    outputs = result.get('outputs', [])
    if outputs:
        print("\n   Output:")
        for output in outputs:
            print(f"   {output}")
else:
    print("   ❌ Execution failed!")
    errors = result.get('errors', [])
    if errors:
        print("\n   Errors:")
        for error in errors:
            print(f"   {error}")

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)
