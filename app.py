"""
Streamlit Frontend - CLEAN REBUILD
Phishing Detection System - Login Based
"""
import streamlit as st
import requests
import os
import sys
from typing import Optional
from pathlib import Path

# Add project root to sys.path for direct imports
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from database.database import get_db
from ml_engine.phishing_detector import PhishingDetector

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL")
# If API_BASE_URL is not set, we use "standalone" mode (direct backend calls)
IS_STANDALONE = API_BASE_URL is None or API_BASE_URL == ""

TIMEOUT = 120

# Initialize direct backend components if standalone
if IS_STANDALONE:
    if 'detector' not in st.session_state:
        st.session_state.detector = PhishingDetector()
    if 'db' not in st.session_state:
        st.session_state.db = get_db()

# Initialize session state
if 'user_logged_in' not in st.session_state:
    st.session_state.user_logged_in = False
    st.session_state.user_id = None
    st.session_state.user_email = None
    st.session_state.user_name = None
    st.session_state.user_role = None
    st.session_state.access_token = None

def check_login():
    """Check if user is logged in"""
    return st.session_state.user_logged_in

def login_page():
    """Login page"""
    st.title("🛡️ Phishing Detection System")
    st.subheader("Login to Your Account")
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            email = st.text_input("📧 Email", placeholder="your@email.com")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
            submit = st.form_submit_button("Login", type="primary", use_container_width=True)
            
            if submit:
                if not email or not password:
                    st.error("⚠️ Please enter both email and password")
                else:
                    with st.spinner("🔐 Logging in..."):
                        try:
                            if IS_STANDALONE:
                                from auth.auth_manager import AuthManager
                                auth_manager = AuthManager(st.session_state.db)
                                result = auth_manager.login_user(email, password)
                            else:
                                response = requests.post(
                                    f"{API_BASE_URL}/api/auth/login",
                                    json={"email": email, "password": password},
                                    timeout=TIMEOUT
                                )
                                if response.status_code == 200:
                                    result = response.json()
                                else:
                                    result = {"success": False, "message": f"API Error: {response.text}"}

                            if result.get('success'):
                                st.session_state.user_logged_in = True
                                user = result.get('user', {})
                                st.session_state.user_id = user.get('id')
                                st.session_state.user_email = user.get('email')
                                st.session_state.user_name = user.get('name', 'User')
                                st.session_state.user_role = user.get('role', 'user')
                                st.session_state.access_token = result.get('token')
                                st.success("✅ Login successful!")
                                st.rerun()
                            else:
                                st.error(f"❌ Login failed: {result.get('message', result.get('error', 'Unknown error'))}")
                        except Exception as e:
                            st.error(f"❌ Error during login: {e}")
        
        st.markdown("---")
        if st.button("📝 Don't have an account? Register here", use_container_width=True):
            st.session_state.show_register = True
            st.rerun()

def register_page():
    """Registration page"""
    st.title("🛡️ Phishing Detection System")
    st.subheader("Create New Account")
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("register_form"):
            name = st.text_input("👤 Name", placeholder="Your full name")
            email = st.text_input("📧 Email", placeholder="your@email.com")
            password = st.text_input("🔒 Password", type="password", placeholder="Create a password")
            confirm_password = st.text_input("🔒 Confirm Password", type="password", placeholder="Confirm password")
            role = st.selectbox("👥 Role", ["user", "admin"], index=0)
            submit = st.form_submit_button("Register", type="primary", use_container_width=True)
            
            if submit:
                if not all([name, email, password, confirm_password]):
                    st.error("⚠️ Please fill all fields")
                elif password != confirm_password:
                    st.error("⚠️ Passwords do not match")
                else:
                    with st.spinner("📝 Registering..."):
                        try:
                            if IS_STANDALONE:
                                from auth.auth_manager import AuthManager
                                auth_manager = AuthManager(st.session_state.db)
                                result = auth_manager.register_user(
                                    name=name, email=email, password=password, role=role
                                )
                            else:
                                response = requests.post(
                                    f"{API_BASE_URL}/api/auth/register",
                                    json={"name": name, "email": email, "password": password, "role": role},
                                    timeout=TIMEOUT
                                )
                                if response.status_code == 200:
                                    result = response.json()
                                else:
                                    result = {"success": False, "message": f"API Error: {response.text}"}

                            if result.get('success'):
                                st.success("✅ Registration successful! Please login.")
                                st.session_state.show_register = False
                                st.rerun()
                            else:
                                st.error(f"❌ Registration failed: {result.get('message', result.get('error', 'Unknown error'))}")
                        except Exception as e:
                            st.error(f"❌ Error during registration: {e}")
        
        st.markdown("---")
        if st.button("🔙 Back to Login", use_container_width=True):
            st.session_state.show_register = False
            st.rerun()

def user_dashboard():
    """User dashboard - Role-based"""
    user_role = st.session_state.user_role or 'user'
    
    # Role-based title and tabs
    if user_role == 'admin':
        st.title(f"🛡️ Admin Portal - Welcome, {st.session_state.user_name}!")
        st.info("👑 You have administrative privileges")
        tabs = st.tabs(["📧 Scan Message", "🔗 Scan URL", "📊 My Scans", "📈 Statistics", "🤖 Auto-Training"])
    elif user_role == 'soc_analyst':
        st.title(f"🛡️ SOC Analyst Portal - Welcome, {st.session_state.user_name}!")
        st.info("🔍 SOC Analyst Dashboard")
        tabs = st.tabs(["📧 Scan Message", "🔗 Scan URL", "📊 All Scans", "📈 Statistics", "🔐 Threat Intelligence"])
    else:
        st.title(f"🛡️ User Portal - Welcome, {st.session_state.user_name}!")
        tabs = st.tabs(["📧 Scan Message", "🔗 Scan URL", "📊 My Scans", "📈 Statistics"])
    
    st.markdown("---")
    
    with tabs[0]:
        st.subheader("Scan Message")
        with st.form("scan_message_form"):
            content = st.text_area("Message Content", height=200, placeholder="Paste email/SMS content here...")
            subject = st.text_input("Subject (optional)")
            channel = st.selectbox("Channel", ["email", "sms", "chat"])
            submit = st.form_submit_button("🔍 Scan Message", type="primary", use_container_width=True)
            
            if submit:
                if not content:
                    st.error("⚠️ Please enter message content")
                else:
                    with st.spinner("🔍 Scanning message..."):
                        try:
                            if IS_STANDALONE:
                                detector = st.session_state.detector
                                db = st.session_state.db
                                
                                # Create message record
                                message_id = db.create_message(
                                    user_id=st.session_state.user_id,
                                    channel=channel,
                                    content=content,
                                    subject=subject
                                )
                                
                                # Run ML detection
                                detection_result = detector.detect_message_phishing(
                                    content=content,
                                    subject=subject
                                )
                                
                                # Update message with prediction
                                db.update_message_prediction(
                                    message_id=message_id,
                                    detected_label=1 if detection_result['is_phishing'] else 0,
                                    confidence_score=detection_result['confidence_score'],
                                    nlp_score=detection_result.get('nlp_score'),
                                    adversarial_score=detection_result.get('adversarial_score'),
                                    risk_factors=detection_result.get('risk_factors', []),
                                    explainable_reasons=detection_result.get('explainable_reasons', {})
                                )
                                result = detection_result
                                result['success'] = True
                            else:
                                response = requests.post(
                                    f"{API_BASE_URL}/api/scan/message",
                                    json={
                                        "content": content,
                                        "subject": subject,
                                        "channel": channel,
                                        "user_id": st.session_state.user_id
                                    },
                                    timeout=TIMEOUT
                                )
                                if response.status_code == 200:
                                    result = response.json()
                                else:
                                    result = {"success": False, "message": f"API Error: {response.text}"}
                            
                            if result.get('success') or 'is_phishing' in result:
                                st.success("✅ Scan completed!")
                                # ... results display logic ...
                                
                                # Display results nicely
                                col1, col2 = st.columns(2)
                                with col1:
                                    if result.get('is_phishing'):
                                        st.error(f"🚨 PHISHING DETECTED! (Risk: {result.get('risk_score', 0):.2%})")
                                    else:
                                        st.success(f"✅ Safe (Risk: {result.get('risk_score', 0):.2%})")
                                
                                with col2:
                                    confidence = result.get('confidence_score', 0)
                                    if confidence is None or confidence == 0:
                                        confidence = result.get('risk_score', 0)
                                    st.metric("Confidence", f"{confidence:.2%}")
                                
                                # Show risk factors
                                if result.get('risk_factors'):
                                    st.warning("⚠️ Risk Factors:")
                                    for factor in result.get('risk_factors', []):
                                        st.write(f"  • {factor}")
                                
                                # Show explainable reasons
                                if result.get('explainable_reasons'):
                                    with st.expander("📊 Detailed Analysis"):
                                        for key, value in result.get('explainable_reasons', {}).items():
                                            st.write(f"**{key.replace('_', ' ').title()}**: {value}")
                                
                                # Show full JSON in expander
                                with st.expander("🔍 Full Results"):
                                    st.json(result)
                            else:
                                st.error(f"❌ Scan failed: {response.text[:200]}")
                        except requests.exceptions.RequestException as e:
                            st.error(f"❌ Error: {e}")
    
    with tabs[1]:
        st.subheader("Scan URL")
        with st.form("scan_url_form"):
            url = st.text_input("URL", placeholder="https://example.com")
            submit = st.form_submit_button("🔍 Scan URL", type="primary", use_container_width=True)
            
            if submit:
                if not url:
                    st.error("⚠️ Please enter a URL")
                else:
                        try:
                            if IS_STANDALONE:
                                detector = st.session_state.detector
                                db = st.session_state.db
                                
                                # Create URL record
                                url_id = db.create_url(raw_url=url)
                                
                                # Run ML detection
                                detection_result = detector.detect_url_phishing(url=url)
                                
                                # Update URL with prediction results
                                db.update_url_prediction(
                                    url_id=url_id,
                                    risk_score=detection_result['risk_score'],
                                    is_phishing=1 if detection_result['is_phishing'] else 0,
                                    gnn_score=detection_result.get('gnn_score'),
                                    cnn_score=detection_result.get('cnn_score'),
                                    redirect_depth=detection_result.get('redirect_depth', 0),
                                    redirect_chain=detection_result.get('redirect_chain', [])
                                )
                                result = detection_result
                                result['success'] = True
                            else:
                                response = requests.post(
                                    f"{API_BASE_URL}/api/scan/url",
                                    json={"url": url, "user_id": st.session_state.user_id},
                                    timeout=TIMEOUT
                                )
                                if response.status_code == 200:
                                    result = response.json()
                                else:
                                    result = {"success": False, "message": f"API Error: {response.text}"}
                            
                            if result.get('success') or 'is_phishing' in result:
                                st.success("✅ Scan completed!")
                                # ... results display logic ...
                                
                                # Display results nicely
                                col1, col2 = st.columns(2)
                                with col1:
                                    if result.get('is_phishing'):
                                        st.error(f"🚨 PHISHING DETECTED! (Risk: {result.get('risk_score', 0):.2%})")
                                    else:
                                        st.success(f"✅ Safe (Risk: {result.get('risk_score', 0):.2%})")
                                
                                with col2:
                                    confidence = result.get('confidence_score', 0)
                                    if confidence is None or confidence == 0:
                                        confidence = result.get('risk_score', 0)
                                    st.metric("Confidence", f"{confidence:.2%}")
                                
                                # Show risk factors
                                if result.get('risk_factors'):
                                    st.warning("⚠️ Risk Factors:")
                                    for factor in result.get('risk_factors', []):
                                        st.write(f"  • {factor}")
                                
                                # Show explainable reasons
                                if result.get('explainable_reasons'):
                                    with st.expander("📊 Detailed Analysis"):
                                        for key, value in result.get('explainable_reasons', {}).items():
                                            st.write(f"**{key.replace('_', ' ').title()}**: {value}")
                                
                                # Show full JSON in expander
                                with st.expander("🔍 Full Results"):
                                    st.json(result)
                            else:
                                st.error(f"❌ Scan failed: {response.text[:200]}")
                        except requests.exceptions.RequestException as e:
                            st.error(f"❌ Error: {e}")
    
    with tabs[2]:
        if user_role == 'soc_analyst':
            st.subheader("All Scans (SOC View)")
            st.info("🔍 Viewing all user scans across the system")
        else:
            st.subheader("My Scans")
        
        scan_type = st.radio("Scan Type", ["Messages", "URLs"], horizontal=True)
        
        try:
            if scan_type == "Messages":
                # SOC analysts can see all messages, users see only their own
                params = {"limit": 100}
                if user_role != 'soc_analyst':
                    params["user_id"] = st.session_state.user_id
                
                if IS_STANDALONE:
                    db = st.session_state.db
                    messages = db.get_user_messages(st.session_state.user_id, limit=100)
                    data = {"success": True, "messages": messages}
                    status_code = 200
                else:
                    response = requests.get(
                        f"{API_BASE_URL}/api/scans/messages",
                        params=params,
                        timeout=TIMEOUT
                    )
                    status_code = response.status_code
                    if status_code == 200:
                        data = response.json()
                
                if status_code == 200:
                    messages = data.get('messages', [])
                    
                    if messages:
                        st.write(f"**Total Scans:** {len(messages)}")
                        st.markdown("---")
                        
                        for msg in messages:
                            with st.expander(f"📧 {msg.get('subject', 'No Subject')} - {msg.get('created_at', '')[:19]}"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    if msg.get('detected_label') == 1:
                                        st.error(f"🚨 PHISHING (Confidence: {msg.get('confidence_score', 0):.2%})")
                                    else:
                                        st.success(f"✅ Safe (Confidence: {1 - msg.get('confidence_score', 0):.2%})")
                                with col2:
                                    st.write(f"**Channel:** {msg.get('channel', 'N/A')}")
                                
                                st.write(f"**Content:** {msg.get('content', '')[:200]}...")
                                if msg.get('sender'):
                                    st.write(f"**From:** {msg.get('sender')}")
                    else:
                        st.info("No message scans found. Start scanning messages to see history here.")
                else:
                    st.error("Failed to load message scans")
            else:
                if IS_STANDALONE:
                    db = st.session_state.db
                    urls = db.get_user_urls(limit=100)
                    data = {"success": True, "urls": urls}
                    status_code = 200
                else:
                    response = requests.get(
                        f"{API_BASE_URL}/api/scans/urls",
                        params={"limit": 100},
                        timeout=TIMEOUT
                    )
                    status_code = response.status_code
                    if status_code == 200:
                        data = response.json()
                
                if status_code == 200:
                    urls = data.get('urls', [])
                    
                    if urls:
                        st.write(f"**Total URL Scans:** {len(urls)}")
                        st.markdown("---")
                        
                        for url_data in urls:
                            with st.expander(f"🔗 {url_data.get('raw_url', 'N/A')[:60]}... - {url_data.get('created_at', '')[:19]}"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    if url_data.get('is_phishing') == 1:
                                        st.error(f"🚨 PHISHING (Risk: {url_data.get('risk_score', 0):.2%})")
                                    else:
                                        st.success(f"✅ Safe (Risk: {url_data.get('risk_score', 0):.2%})")
                                with col2:
                                    st.write(f"**GNN Score:** {url_data.get('gnn_score', 0):.2%}")
                                    st.write(f"**CNN Score:** {url_data.get('cnn_score', 0):.2%}")
                                
                                st.write(f"**URL:** {url_data.get('raw_url', 'N/A')}")
                    else:
                        st.info("No URL scans found. Start scanning URLs to see history here.")
                else:
                    st.error("Failed to load URL scans")
        except Exception as e:
            st.error(f"Error loading scans: {e}")
    
    # Statistics Tab
    with tabs[3]:
        st.subheader("Statistics")
        
        # Statistics
        try:
            if IS_STANDALONE:
                db = st.session_state.db
                stats = db.get_statistics()
                status_code = 200
            else:
                response = requests.get(f"{API_BASE_URL}/api/statistics", timeout=TIMEOUT)
                status_code = response.status_code
                if status_code == 200:
                    stats = response.json()
            
            if status_code == 200:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Messages", stats.get('total_messages', 0))
                    st.metric("Phishing Messages", stats.get('phishing_messages', 0))
                with col2:
                    st.metric("Total URLs", stats.get('total_urls', 0))
                    st.metric("Phishing URLs", stats.get('phishing_urls', 0))
                
                # Calculate percentages
                if stats.get('total_messages', 0) > 0:
                    phishing_msg_pct = (stats.get('phishing_messages', 0) / stats.get('total_messages', 1)) * 100
                    st.metric("Phishing Message Rate", f"{phishing_msg_pct:.1f}%")
                
                if stats.get('total_urls', 0) > 0:
                    phishing_url_pct = (stats.get('phishing_urls', 0) / stats.get('total_urls', 1)) * 100
                    st.metric("Phishing URL Rate", f"{phishing_url_pct:.1f}%")
            else:
                st.error("Failed to load statistics")
        except Exception as e:
            st.error(f"Error: {e}")
    
    # Auto-Training Tab (Admin only)
    if user_role == 'admin' and len(tabs) > 4:
        with tabs[4]:
            st.subheader("🤖 Auto-Training")
            st.markdown("### Model Training & Management")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **What is Auto-Training?**
                - Trains ML models using CSV data + Database feedback
                - Improves detection accuracy over time
                - Creates new model versions automatically
                """)
            
            with col2:
                st.markdown("""
                **Training Data Sources:**
                - Historical CSV datasets
                - User feedback from scans
                - Incremental learning enabled
                """)
            
            st.markdown("---")
            
            # Training Controls
            st.markdown("### 🚀 Start Training")
            
            model_type = st.selectbox("Select Model Type", ["nlp", "cnn", "gnn"], index=0)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                start_training = st.button("🚀 Start Auto-Training", type="primary", use_container_width=True)
            with col2:
                check_status = st.button("📊 Check Training Status", use_container_width=True)
            with col3:
                view_models = st.button("📋 View Model Versions", use_container_width=True)
            
            if start_training:
                with st.spinner("🔄 Training models... This may take a few minutes."):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/api/train/start",
                            json={"model_type": model_type},
                            timeout=300
                        )
                        if response.status_code == 200:
                            result = response.json()
                            
                            if isinstance(result, dict) and result.get('success'):
                                st.success("✅ Training completed successfully!")
                                st.balloons()
                                
                                # Show metrics - handle different result structures
                                training_result = result.get('result', {})
                                
                                # If result is a dict with model results
                                if isinstance(training_result, dict):
                                    # Check if it's train_all_models result (dict of model results)
                                    if all(isinstance(v, dict) for v in training_result.values()):
                                        for model_type_key, model_result in training_result.items():
                                            if isinstance(model_result, dict) and model_result.get('success'):
                                                st.markdown(f"### {model_type_key.upper()} Model")
                                                metrics = model_result.get('metrics', {})
                                                if isinstance(metrics, dict):
                                                    col1, col2, col3, col4 = st.columns(4)
                                                    with col1:
                                                        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
                                                    with col2:
                                                        st.metric("Precision", f"{metrics.get('precision', 0):.2%}")
                                                    with col3:
                                                        st.metric("Recall", f"{metrics.get('recall', 0):.2%}")
                                                    with col4:
                                                        st.metric("F1-Score", f"{metrics.get('f1_score', 0):.2%}")
                                    # If it's a single model result
                                    elif training_result.get('success'):
                                        metrics = training_result.get('metrics', {})
                                        if isinstance(metrics, dict):
                                            col1, col2, col3, col4 = st.columns(4)
                                            with col1:
                                                st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
                                            with col2:
                                                st.metric("Precision", f"{metrics.get('precision', 0):.2%}")
                                            with col3:
                                                st.metric("Recall", f"{metrics.get('recall', 0):.2%}")
                                            with col4:
                                                st.metric("F1-Score", f"{metrics.get('f1_score', 0):.2%}")
                                
                                # Show full result in expander
                                with st.expander("🔍 Full Training Result"):
                                    st.json(result)
                            else:
                                error_msg = result.get('error', result.get('detail', 'Unknown error')) if isinstance(result, dict) else str(result)
                                st.error(f"Training failed: {error_msg}")
                        else:
                            st.error(f"Training failed: {response.text[:500]}")
                    except requests.exceptions.Timeout:
                        st.error("⏱️ Training timeout. The process may still be running in the background.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            if check_status:
                st.markdown("### 📊 Training Status")
                try:
                    response = requests.get(f"{API_BASE_URL}/api/train/status", timeout=TIMEOUT)
                    if response.status_code == 200:
                        status_data = response.json()
                        batches = status_data.get('batches', [])
                        
                        if batches:
                            st.write(f"**Total Training Batches:** {len(batches)}")
                            st.markdown("---")
                            
                            for batch in batches[:10]:  # Show last 10
                                status_color = {
                                    'completed': '✅',
                                    'training': '🔄',
                                    'failed': '❌',
                                    'pending': '⏳'
                                }.get(batch.get('status', 'pending'), '⏳')
                                
                                with st.expander(f"{status_color} {batch.get('model_type', 'N/A').upper()} - {batch.get('status', 'N/A')} ({batch.get('created_at', '')[:19]})"):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write(f"**Batch ID:** {batch.get('id')}")
                                        st.write(f"**Type:** {batch.get('batch_type')}")
                                        st.write(f"**Samples:** {batch.get('samples_count', 0)}")
                                    with col2:
                                        st.write(f"**Status:** {batch.get('status')}")
                                        if batch.get('accuracy_before'):
                                            st.write(f"**Accuracy Before:** {batch.get('accuracy_before'):.2%}")
                                        if batch.get('accuracy_after'):
                                            st.write(f"**Accuracy After:** {batch.get('accuracy_after'):.2%}")
                        else:
                            st.info("No training batches found.")
                    else:
                        st.error("Failed to load training status")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            
            if view_models:
                st.markdown("### 📋 Model Versions")
                try:
                    response = requests.get(f"{API_BASE_URL}/api/train/models", timeout=TIMEOUT)
                    if response.status_code == 200:
                        models_data = response.json()
                        versions = models_data.get('versions', [])
                        
                        if versions:
                            st.write(f"**Total Model Versions:** {len(versions)}")
                            st.markdown("---")
                            
                            # Group by model type
                            model_types = {}
                            for version in versions:
                                model_type = version.get('model_type', 'unknown')
                                if model_type not in model_types:
                                    model_types[model_type] = []
                                model_types[model_type].append(version)
                            
                            for model_type, type_versions in model_types.items():
                                st.markdown(f"#### {model_type.upper()} Models")
                                
                                for version in sorted(type_versions, key=lambda x: x.get('created_at', ''), reverse=True)[:5]:
                                    is_active = version.get('is_active', 0) == 1
                                    active_badge = "🟢 ACTIVE" if is_active else "⚪ Inactive"
                                    
                                    with st.expander(f"{active_badge} {version.get('version', 'N/A')} - {version.get('created_at', '')[:19]}"):
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.write(f"**Version:** {version.get('version')}")
                                            st.write(f"**Model Type:** {version.get('model_type')}")
                                            st.write(f"**Training Samples:** {version.get('training_samples', 0)}")
                                            st.write(f"**Model Path:** {version.get('model_path', 'N/A')}")
                                        with col2:
                                            if version.get('accuracy'):
                                                st.metric("Accuracy", f"{version.get('accuracy'):.2%}")
                                            if version.get('precision'):
                                                st.metric("Precision", f"{version.get('precision'):.2%}")
                                            if version.get('recall'):
                                                st.metric("Recall", f"{version.get('recall'):.2%}")
                                            if version.get('f1_score'):
                                                st.metric("F1-Score", f"{version.get('f1_score'):.2%}")
                                
                                st.markdown("---")
                        else:
                            st.info("No model versions found.")
                    else:
                        st.error("Failed to load model versions")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            
            st.markdown("---")
            st.markdown("### ⚠️ Important Notes")
            st.warning("""
            - Training may take 5-15 minutes depending on data size
            - Models are saved automatically after training
            - New model versions are created with timestamps
            - Active models are automatically updated
            """)
    
    # SOC Analyst Tab (SOC Analyst only)
    if user_role == 'soc_analyst' and len(tabs) > 4:
        with tabs[4]:
            st.subheader("🔐 Threat Intelligence")
            st.info("SOC Analyst features coming soon!")
            st.markdown("""
            **Planned Features:**
            - Threat intelligence feeds
            - IOC management
            - Advanced analytics
            - Incident response tools
            """)
    
    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.user_logged_in = False
        st.session_state.user_id = None
        st.session_state.user_email = None
        st.session_state.user_name = None
        st.session_state.user_role = None
        st.session_state.access_token = None
        st.rerun()

def main():
    """Main application"""
    if not check_login():
        if st.session_state.get('show_register', False):
            register_page()
        else:
            login_page()
    else:
        user_dashboard()

if __name__ == "__main__":
    main()

```
