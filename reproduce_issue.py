import requests
import json

def test_url(url):
    print(f"\nTesting URL: {url}")
    try:
        response = requests.post(
            "http://localhost:8000/api/scan/url",
            json={"url": url},
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            print(f"Result: {'PHISHING' if result['is_phishing'] else 'SAFE'}")
            print(f"Confidence: {result['confidence_score']}")
            print(f"Risk factors: {result['risk_factors']}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Connection error: {e}")

def test_message(content, subject=None):
    print(f"\nTesting Message: {content[:50]}...")
    try:
        response = requests.post(
            "http://localhost:8000/api/scan/message",
            json={"content": content, "subject": subject},
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            print(f"Result: {'PHISHING' if result['is_phishing'] else 'SAFE'}")
            print(f"Confidence: {result['confidence_score']}")
            print(f"Risk factors: {result['risk_factors']}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Connection error: {e}")

if __name__ == "__main__":
    # Test URLs
    test_url("http://secure-login-paypal.com/verify") # Fake link
    test_url("https://www.google.com") # Safe link
    
    # Test Messages
    test_message("URGENT: Your account has been locked. Please login here to verify: http://evil.com/login", "Security Alert")
    test_message("Hey, let's meet for coffee tomorrow at 10am.", "Coffee?")
