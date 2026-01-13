@echo off
echo ============================================================
echo 🛡️  PHISHING DETECTION SYSTEM - SETUP SCRIPT
echo ============================================================
echo.

REM 1. Create Virtual Environment
echo [1/3] Creating Virtual Environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment. Ensure Python is installed.
    pause
    exit /b %errorlevel%
)

REM 2. Install Dependencies
echo [2/3] Installing Dependencies (This may take a few minutes)...
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Dependency installation failed.
    pause
    exit /b %errorlevel%
)

REM 3. Initialize Database & Default Admin
echo [3/3] Initializing System & Creating Default Admin...
python create_default_users.py
if %errorlevel% neq 0 (
    echo [ERROR] Database initialization failed.
    pause
    exit /b %errorlevel%
)

echo.
echo ============================================================
echo ✅ SETUP COMPLETED SUCCESSFULLY!
echo ============================================================
echo.
echo To run the project:
echo 1. Run 'run_project.bat' to launch the System
echo.
pause
