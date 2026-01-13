import requests

def test_single_url(url):
    print(f"Testing URL: {url}")
    try:
        response = requests.post("http://localhost:8000/api/scan/url", json={"url": url}, timeout=30)
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

def test_single_msg(msg):
    print(f"Testing MSG: {msg}")
    try:
        response = requests.post("http://localhost:8000/api/scan/message", json={"content": msg}, timeout=30)
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_single_url("http://secure-login-paypal.com/verify")
    test_single_msg("URGENT: Your account is locked. Login at http://evil.com")
