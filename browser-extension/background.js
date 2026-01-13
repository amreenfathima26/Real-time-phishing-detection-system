// Background service worker for Real-Time Phishing Detection Extension

const API_BASE_URL = 'http://localhost:8000'; // Change to your API URL

// Listen for tab updates
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && tab.url) {
    checkURL(tab.url, tabId);
  }
});

// Check URL for phishing
async function checkURL(url, tabId) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/scan/url`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ url: url })
    });

    const result = await response.json();

    if (result.is_phishing) {
      // Show warning
      showWarning(tabId, result);
      
      // Store result
      chrome.storage.local.set({
        [`phishing_${tabId}`]: {
          url: url,
          result: result,
          timestamp: Date.now()
        }
      });
    }
  } catch (error) {
    console.error('Error checking URL:', error);
  }
}

// Show warning to user
function showWarning(tabId, result) {
  chrome.tabs.sendMessage(tabId, {
    action: 'showWarning',
    data: result
  });
}

// Listen for messages from content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'checkURL') {
    checkURL(request.url, sender.tab.id).then(sendResponse);
    return true; // Keep channel open for async response
  }
  
  if (request.action === 'submitFeedback') {
    submitFeedback(request.feedback).then(sendResponse);
    return true;
  }
});

// Submit feedback
async function submitFeedback(feedback) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/feedback`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(feedback)
    });

    return await response.json();
  } catch (error) {
    console.error('Error submitting feedback:', error);
    return { success: false, error: error.message };
  }
}

// Intercept web requests (optional - for advanced detection)
chrome.webRequest.onBeforeRequest.addListener(
  (details) => {
    // Can add additional checks here
    return { cancel: false };
  },
  { urls: ["<all_urls>"] },
  ["blocking"]
);

