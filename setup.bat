@echo off
echo ===============================
echo EMOTION DETECTION WEB APP SETUP (by Toluwanimi)
echo ===============================

:: Create virtual environment
python -m venv venv

:: Activate virtual environment
call venv\Scripts\activate

:: Upgrade pip
python -m pip install --upgrade pip

:: Install dependencies
pip install -r requirements.txt

echo.
echo ===============================
echo ✅ Setup Complete!
echo To start your app, type:
echo venv\Scripts\activate
echo python app.py
echo ===============================
pause
