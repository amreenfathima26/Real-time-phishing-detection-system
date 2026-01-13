@echo off
echo ============================================================
echo 🚀 STARTING PHISHING DETECTION SYSTEM...
echo ============================================================
echo.

REM Activate virtual environment if it exists
if exist venv\Scripts\activate (
    echo [OK] Activating virtual environment...
    call venv\Scripts\activate
) else (
    echo [!] Virtual environment not found. Running with system python...
)

REM Start Backend in a new window
echo [OK] Starting Backend API (Port 8000)...
start /B cmd /c "cd backend && python -m uvicorn main:app --host 127.0.0.1 --port 8000"

REM Wait a few seconds for backend to initialize
timeout /t 5 /nobreak > nul

REM Start Frontend
echo [OK] Starting Frontend Dashboard (Port 8501)...
streamlit run app.py --server.port 8501

echo.
echo [INFO] System is running. Close this window to stop.
pause
