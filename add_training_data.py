import pandas as pd
import sqlite3
import os
from pathlib import Path

def add_url_data():
    csv_path = 'data/urls_training.csv'
    new_data = [
        # Phishing URLs
        ['http://secure-login-wellsfargo.com', 1],
        ['http://account-update-amazon.net', 1],
        ['http://chase-online-security.org', 1],
        ['http://verify-identity-apple.com', 1],
        ['http://paypal-resolution-center.com', 1],
        ['http://netflix-billing-update.com', 1],
        ['http://microsoft-account-locked.xyz', 1],
        ['http://google-drive-shared-file.cf', 1],
        ['http://facebook-security-alert.ml', 1],
        ['http://bankamerica-enrollment.ga', 1],
        ['http://irs-tax-refund-status.online', 1],
        ['http://post-office-delivery-failed.com', 1],
        ['http://bitcoin-wallet-recovery.io', 1],
        ['http://steam-community-giveaway.ru', 1],
        ['http://discord-nitro-free.com', 1],
        # Legitimate URLs
        ['https://www.google.com', 0],
        ['https://www.github.com', 0],
        ['https://www.microsoft.com', 0],
        ['https://www.apple.com', 0],
        ['https://www.amazon.com', 0],
        ['https://www.netflix.com', 0],
        ['https://www.facebook.com', 0],
        ['https://www.linkedin.com', 0],
        ['https://www.twitter.com', 0],
        ['https://www.wikipedia.org', 0],
        ['https://www.reddit.com', 0],
        ['https://www.stackoverflow.com', 0],
        ['https://www.medium.com', 0],
        ['https://www.youtube.com', 0],
        ['https://www.nytimes.com', 0]
    ]
    
    df_new = pd.DataFrame(new_data, columns=['url', 'is_phishing'])
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df_combined = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates()
        df_combined.to_csv(csv_path, index=False)
        print(f"✅ Added {len(df_combined) - len(df_old)} new URLs to {csv_path}")
    else:
        df_new.to_csv(csv_path, index=False)
        print(f"✅ Created {csv_path} with {len(df_new)} URLs")

def add_message_data():
    csv_path = 'data/phishing_messages.csv'
    new_data = [
        # Phishing Messages
        ['Your PayPal account has been limited. Please verify your identity immediately to restore access.', 'Action Required: Account Limited', 1],
        ['Urgent: Suspicious activity detected on your Amazon account. Click here to secure your account: http://bit.ly/secure-amz', 'Security Alert', 1],
        ['Dear customer, your Chase debit card has been suspended. Call us at 1-800-PHISH to reactive.', 'Important Notice', 1],
        ['IRS: Your tax refund of $1,200.50 is ready. Claim it now at http://irs-refund-gov.com', 'Refund Status', 1],
        ['Your Apple ID was used to sign in to a new device in Moscow, Russia. If this wasn\'t you, reset your password now.', 'Sign-in Alert', 1],
        ['Congratulations! You have won a $500 Walmart gift card. Click here to claim: http://win-walmart.com', 'Winner Announcement', 1],
        ['Netflix: Your payment method has expired. Update it now to avoid service interruption: http://netflix-update.com', 'Payment Failed', 1],
        ['Microsoft: Someone tried to access your Outlook account. Please verify your account at http://outlook-verify.com', 'Security Warning', 1],
        # Legitimate Messages
        ['Hey, are we still meeting for lunch today at 12?', 'Lunch?', 0],
        ['The project deadline has been extended to Friday. Please update your status reports.', 'Project Update', 0],
        ['Your package from Amazon has been delivered. See details at https://amazon.com/orders', 'Delivery Update', 0],
        ['Your one-time passcode for login is 123456. Do not share this with anyone.', 'Verification Code', 0],
        ['Don\'t forget our meeting at 3 PM today in the conference room.', 'Reminder', 0],
        ['Thanks for subscribing to our newsletter! You will receive weekly updates.', 'Welcome!', 0],
        ['Your monthly bank statement is now available for viewing on our secure portal.', 'Statement Available', 0]
    ]
    
    df_new = pd.DataFrame(new_data, columns=['content', 'subject', 'is_phishing'])
    df_new.to_csv(csv_path, index=False)
    print(f"✅ Created {csv_path} with {len(df_new)} messages")

def inject_db_data():
    db_path = 'phishing_detection.db'
    if not os.path.exists(db_path):
        print("❌ Database not found")
        return
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Add URLs
    urls = [
        ('http://login.security-update.com', 1),
        ('http://microsoft-verify.net', 1),
        ('https://www.google.com', 0),
        ('https://www.github.com', 0)
    ]
    for url, is_p in urls:
        cursor.execute("INSERT OR IGNORE INTO urls (raw_url, is_phishing) VALUES (?, ?)", (url, is_p))
        
    # Add Messages
    messages = [
        ('Your bank account will be closed in 24 hours. Verify now: http://bank-verify.com', 'Final Warning', 1),
        ('Did you see the new movie that came out last night?', 'Movie Night', 0)
    ]
    for msg, subj, is_p in messages:
        cursor.execute("INSERT OR IGNORE INTO messages (content, subject, detected_label) VALUES (?, ?, ?)", (msg, subj, is_p))
        
    conn.commit()
    conn.close()
    print("✅ Injected data into SQLite database")

if __name__ == "__main__":
    add_url_data()
    add_message_data()
    inject_db_data()
