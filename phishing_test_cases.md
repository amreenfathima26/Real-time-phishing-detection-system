# Comprehensive Phishing Test Suite

This file contains 20+ test cases (URLs and Messages) designed to verify the "Ultra-Robust" phishing detection system.

## 🚨 Phishing URLs (High Risk)

| # | Type | URL | Description |
| :--- | :--- | :--- | :--- |
| 1 | **Homograph** | `http://p4ypal-security.net` | Number '4' substituted for 'a' |
| 2 | **Homograph** | `http://paypa1-update.com` | Number '1' substituted for 'l' |
| 3 | **Hijacking** | `http://paypal.com.secure-login.xyz` | Brand as prefix on a rogue domain |
| 4 | **Hijacking** | `http://microsoft.com.account-verify.tk` | Brand prefix with suspicious TLD |
| 5 | **Structural** | `https://secure.example/auth/login/session/verify/identity` | Deep path with high-risk keywords |
| 6 | **IP Based** | `http://192.168.1.1/login.php` | Direct IP usage (very suspicious in public mail) |
| 7 | **Typosquat** | `http://amz-onn.site` | Deliberate misspelling and extra hyphens |
| 8 | **Subdomain** | `http://verify.chase.com.login-portal.co` | Complex subdomain nesting |
| 9 | **Lookalike** | `http://facetook.com` | Visual similarity ('t' for 'b') |
| 10 | **Urgency** | `http://urgent-action-required.com/verify` | Keyword-heavy domain |

## 🚨 Phishing Messages (High Risk)

| # | Subject | Message Content |
| :--- | :--- | :--- |
| 11 | **Security Alert** | "Immediate Action Required: Your account has been suspended due to unauthorized access. Verify now: http://p4ypal-security.net" |
| 12 | **Payment Failed** | "We couldn't process your last payment. To avoid service disruption, update your billing details here: http://paypa1-update.com" |
| 13 | **Security Warning** | "New login detected from Russia. If this wasn't you, secure your account at: http://192.168.1.105/secure" |
| 14 | **Tax Refund** | "You have an outstanding tax refund of $450.20. Claim your refund by logging into the portal: http://irs-refund-claim.org" |
| 15 | **Cloud Update** | "Your cloud storage is full. Increase your limit for free by clicking this link: http://google.com.storage-increase.xyz" |

## ✅ Legitimate Cases (Zero/Low Risk)

| # | Type | Input | Expected Result |
| :--- | :--- | :--- | :--- |
| 16 | **Safe URL** | `https://www.google.com` | **SAFE** |
| 17 | **Safe URL** | `https://github.com/trending` | **SAFE** |
| 18 | **Safe URL** | `https://www.paypal.com/signin` | **SAFE** |
| 19 | **Safe Msg** | "Hey, are you coming to the meeting at 3 PM today?" | **SAFE** |
| 20 | **Safe Msg** | "The report you requested is attached. Best regards, HR Team." | **SAFE** |
| 21 | **Safe Msg** | "Thanks for the coffee yesterday. Let's sync up tomorrow." | **SAFE** |

---
**Usage Instructions:**
1. Copy the URL or Message text.
2. Paste into the **URL Phishing Scanner** or **Message Analyzer** in the dashboard.
3. Verify that the "Ultra-Robust" logic correctly flags the risk factors.
