// Content script for Real-Time Phishing Detection Extension

// Listen for messages from background script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'showWarning') {
    showPhishingWarning(request.data);
  }
});

// Show phishing warning overlay
function showPhishingWarning(result) {
  // Remove existing warning if any
  const existingWarning = document.getElementById('phishing-warning-overlay');
  if (existingWarning) {
    existingWarning.remove();
  }

  // Create warning overlay
  const overlay = document.createElement('div');
  overlay.id = 'phishing-warning-overlay';
  overlay.style.cssText = `
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.7);
    z-index: 999999;
    display: flex;
    justify-content: center;
    align-items: center;
    font-family: Arial, sans-serif;
  `;

  const warningBox = document.createElement('div');
  warningBox.style.cssText = `
    background: white;
    padding: 30px;
    border-radius: 10px;
    max-width: 500px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
  `;

  warningBox.innerHTML = `
    <div style="text-align: center; margin-bottom: 20px;">
      <h2 style="color: #d32f2f; margin: 0 0 10px 0;">⚠️ Phishing Warning</h2>
      <p style="color: #666; margin: 0;">This website has been flagged as potentially dangerous</p>
    </div>
    
    <div style="background: #ffebee; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
      <p style="margin: 0 0 10px 0;"><strong>Risk Score:</strong> ${(result.risk_score * 100).toFixed(1)}%</p>
      <p style="margin: 0 0 10px 0;"><strong>Confidence:</strong> ${(result.confidence_score * 100).toFixed(1)}%</p>
      ${result.risk_factors.length > 0 ? `
        <p style="margin: 10px 0 5px 0;"><strong>Risk Factors:</strong></p>
        <ul style="margin: 0; padding-left: 20px;">
          ${result.risk_factors.map(factor => `<li>${factor}</li>`).join('')}
        </ul>
      ` : ''}
    </div>
    
    <div style="display: flex; gap: 10px;">
      <button id="phishing-warning-back" style="
        flex: 1;
        padding: 10px;
        background: #1976d2;
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-size: 14px;
      ">Go Back</button>
      <button id="phishing-warning-continue" style="
        flex: 1;
        padding: 10px;
        background: #666;
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-size: 14px;
      ">Continue Anyway</button>
      <button id="phishing-warning-report" style="
        flex: 1;
        padding: 10px;
        background: #d32f2f;
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-size: 14px;
      ">Report</button>
    </div>
  `;

  overlay.appendChild(warningBox);
  document.body.appendChild(overlay);

  // Button handlers
  document.getElementById('phishing-warning-back').addEventListener('click', () => {
    window.history.back();
    overlay.remove();
  });

  document.getElementById('phishing-warning-continue').addEventListener('click', () => {
    overlay.remove();
  });

  document.getElementById('phishing-warning-report').addEventListener('click', () => {
    // Submit feedback
    chrome.runtime.sendMessage({
      action: 'submitFeedback',
      feedback: {
        item_type: 'url',
        item_id: result.item_id,
        feedback_type: 'true_positive',
        correct_label: 1
      }
    });
    
    alert('Thank you for reporting! This helps improve our detection system.');
    overlay.remove();
  });
}

// Check links on page
function checkLinks() {
  const links = document.querySelectorAll('a[href]');
  links.forEach(link => {
    link.addEventListener('click', (e) => {
      const url = link.href;
      if (url && url.startsWith('http')) {
        chrome.runtime.sendMessage({
          action: 'checkURL',
          url: url
        });
      }
    });
  });
}

// Initialize
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', checkLinks);
} else {
  checkLinks();
}

