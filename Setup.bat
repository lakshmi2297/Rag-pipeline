@echo off
echo Setting up RAG Pipeline...

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found! Please install from python.org
    pause
    exit /b 1
)

echo Python detected successfully

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo Installing required packages...
pip install -r requirements.txt

REM Create environment file if it doesn't exist
if not exist ".env" (
    copy .env.example .env
    echo Created .env file - please edit with your Mistral API key
)

echo Setup completed successfully!
echo Run start.bat to launch the application
pause
