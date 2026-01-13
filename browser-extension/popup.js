// Popup script for Real-Time Phishing Detection Extension

const API_BASE_URL = 'http://localhost:8000';

// Load stats
async function loadStats() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/statistics`);
    const stats = await response.json();
    
    document.getElementById('scanned-count').textContent = 
      (stats.total_messages || 0) + (stats.total_urls || 0);
    document.getElementById('blocked-count').textContent = 
      (stats.phishing_messages || 0) + (stats.phishing_urls || 0);
  } catch (error) {
    console.error('Error loading stats:', error);
  }
}

// Scan current page
document.getElementById('scan-current').addEventListener('click', async () => {
  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    if (tabs[0] && tabs[0].url) {
      try {
        const response = await fetch(`${API_BASE_URL}/api/scan/url`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: tabs[0].url })
        });
        
        const result = await response.json();
        
        if (result.is_phishing) {
          alert(`⚠️ Phishing Detected!\nRisk Score: ${(result.risk_score * 100).toFixed(1)}%`);
        } else {
          alert(`✅ Safe URL\nRisk Score: ${(result.risk_score * 100).toFixed(1)}%`);
        }
      } catch (error) {
        alert('Error scanning URL: ' + error.message);
      }
    }
  });
});

// Open dashboard
document.getElementById('open-dashboard').addEventListener('click', () => {
  chrome.tabs.create({ url: `${API_BASE_URL}/dashboard` });
});

// Initialize
loadStats();

