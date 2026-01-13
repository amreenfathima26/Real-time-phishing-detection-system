import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from ml_engine.phishing_detector import PhishingDetector

def run_diagnostic():
    detector = PhishingDetector()
    
    test_cases = [
        # Phishing URLs
        {"type": "url", "input": "https://secure.example/login/auth/redirect/session/verify"},
        {"type": "url", "input": "http://p4ypal-login.com"},
        {"type": "url", "input": "http://google.com.secure-login.xyz"},
        {"type": "url", "input": "http://paypa1-update.biz"},
        
        # Legitimate URLs (False Positive Check)
        {"type": "url", "input": "https://www.google.com"},
        {"type": "url", "input": "https://www.microsoft.com"},
        {"type": "url", "input": "https://www.paypal.com"},
        {"type": "url", "input": "https://github.com/trending"},
        
        # Messages
        {"type": "message", "content": "Dear customer, your account is suspended. Verify now: http://evil.com", "subject": "Security Alert"},
        {"type": "message", "content": "Hey, see you at 8?", "subject": "Hi"},
        {"type": "message", "content": "The meeting is at 10am tomorrow in the conference room.", "subject": "Sync"},
    ]
    
    print("\n" + "="*50)
    print("PHISHING DETECTION DIAGNOSTIC")
    print("="*50)
    
    results = []
    for case in test_cases:
        if case["type"] == "url":
            res = detector.detect_url_phishing(case["input"])
            print(f"\nURL: {case['input']}")
            print(f"  Result: {'PHISHING' if res['is_phishing'] else 'SAFE'}")
            print(f"  Confidence: {res['confidence_score']:.4f}")
            print(f"  Risk Factors: {res['risk_factors']}")
            results.append(f"URL: {case['input']} | Result: {'PHISHING' if res['is_phishing'] else 'SAFE'} | Score: {res['confidence_score']:.4f}")
        else:
            res = detector.detect_message_phishing(case["content"], case.get("subject"))
            print(f"\nMSG: {case['content'][:50]}...")
            print(f"  Result: {'PHISHING' if res['is_phishing'] else 'SAFE'}")
            print(f"  Confidence: {res['confidence_score']:.4f}")
            print(f"  Risk Factors: {res['risk_factors']}")
            results.append(f"MSG: {case['content'][:30]}... | Result: {'PHISHING' if res['is_phishing'] else 'SAFE'} | Score: {res['confidence_score']:.4f}")

    with open("verification_results.txt", "w") as f:
        f.write("=== FINAL VERIFICATION RESULTS ===\n")
        for r in results:
            f.write(r + "\n")

if __name__ == "__main__":
    run_diagnostic()
