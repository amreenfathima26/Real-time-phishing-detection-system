import pandas as pd
import os

def generate_perfect_separation():
    # 1. URL DATA
    urls_path = 'data/urls_training.csv'
    os.makedirs('data', exist_ok=True)
    
    new_urls = []
    # Phishing: always has 'verify' or 'secure' or 'login' + brand
    for i in range(500):
        brand = ["paypal", "amazon", "apple", "google", "microsoft", "bank"][i % 6]
        action = ["verify", "secure", "login", "update", "locked"][i % 5]
        new_urls.append([f"http://{brand}-{action}-account-details-{i}.com", 1])
        
    # Legitimate: always contains /about, /contact, or /support and NO suspicious words
    for i in range(500):
        domain = ["google.com", "github.com", "microsoft.com", "apple.com", "wikipedia.org"][i % 5]
        path = ["about", "contact", "support", "index", "legal"][i % 5]
        new_urls.append([f"https://www.{domain}/{path}-{i}", 0])

    pd.DataFrame(new_urls, columns=['url', 'is_phishing']).to_csv(urls_path, index=False)
    print(f"✅ Injected {len(new_urls)} URLs into {urls_path}")

    # 2. MESSAGE DATA
    msgs_path = 'data/phishing_messages.csv'
    phish = []
    for i in range(500):
        brand = ["PAYPAL", "AMAZON", "APPLE", "GOOGLE", "MICROSOFT", "CHASE"][i % 6]
        phish.append([f"URGENT: Your {brand} account is suspended. Please verify now: http://{brand.lower()}-verify.com/{i}", "Security Alert", 1])
        
    legit = []
    for i in range(500):
        name = ["John", "Sarah", "Mike", "Emily", "David"][i % 5]
        legit.append([f"Hi {name}, thanks for the update. Let's meet tomorrow for the sync conference. Best regards!", "Update", 0])

    pd.DataFrame(phish + legit, columns=['content', 'subject', 'is_phishing']).to_csv(msgs_path, index=False)
    print(f"✅ Injected {len(phish)+len(legit)} messages into {msgs_path}")

if __name__ == "__main__":
    generate_perfect_separation()
