import httpx
import base64
import asyncio

# Your GigaChat credentials
GIGACHAT_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1"
GIGACHAT_TOKEN = "MTk3YTc2NDItY2E5OS00MmNkLWE2YTQtMTIwNTFhMDUyZWVkOjMwMDFjYzU2LThiOTMtNDgwOC1iYjQ1LWZjMWU5NzFkZjM1OQ=="

async def test_authentication():
    # Create an async HTTP client
    client = httpx.AsyncClient(timeout=httpx.Timeout(15), verify=False)
    
    try:
        # Method 1: Try using the token as a Bearer token directly
        print("Testing Method 1: Bearer token...")
        url = f"{GIGACHAT_BASE_URL}/models"  # Simple endpoint to test authentication
        headers = {
            "Authorization": f"Bearer {GIGACHAT_TOKEN}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        response = await client.get(url, headers=headers)
        print(f"Method 1 Status: {response.status_code}")
        print(f"Method 1 Response: {response.text}")
        
        # Method 2: Try using the token as Basic Auth
        print("\nTesting Method 2: Basic Auth...")
        headers = {
            "Authorization": f"Basic {GIGACHAT_TOKEN}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        response = await client.get(url, headers=headers)
        print(f"Method 2 Status: {response.status_code}")
        print(f"Method 2 Response: {response.text}")
        
        # Method 3: Try to get an access token first
        print("\nTesting Method 3: OAuth2 client credentials flow...")
        token_url = f"{GIGACHAT_BASE_URL}/token"
        headers = {
            "Authorization": f"Basic {GIGACHAT_TOKEN}",
            "Content-Type": "application/x-www-form-urlencoded",
            "RqUID": "test-request-id",
        }
        data = {
            "grant_type": "client_credentials",
        }
        
        response = await client.post(token_url, headers=headers, data=data)
        print(f"Method 3 Status: {response.status_code}")
        print(f"Method 3 Response: {response.text}")
        
        if response.status_code == 200:
            token_data = response.json()
            access_token = token_data.get("access_token")
            if access_token:
                print("\nTesting with obtained access token...")
                headers = {
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                }
                
                response = await client.get(url, headers=headers)
                print(f"Access Token Status: {response.status_code}")
                print(f"Access Token Response: {response.text}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.aclose()

if __name__ == "__main__":
    asyncio.run(test_authentication())
