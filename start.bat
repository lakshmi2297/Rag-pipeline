@echo off
echo Starting RAG Pipeline API...

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found. Please run setup.bat first.
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

echo Server starting...
echo API will be available at: http://127.0.0.1:8000
echo Interactive docs at: http://127.0.0.1:8000/docs

REM Start the application
python run.py

pause
