@echo off
REM SpeechLab Setup Script for Windows
REM Creates Python venv and installs all dependencies

echo.
echo ========================================
echo   SpeechLab Setup Script
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    exit /b 1
)

REM Create Python virtual environment
echo [1/5] Creating Python virtual environment...
if exist ".venv" (
    echo       Virtual environment already exists, skipping...
) else (
    python -m venv .venv
    echo       Created .venv directory
)

REM Activate venv and install dependencies
echo [2/5] Installing Python dependencies...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -e ".[dev]"

REM Copy environment file
echo [3/5] Setting up environment file...
if not exist ".env" (
    copy .env.example .env
    echo       Created .env from .env.example
) else (
    echo       .env already exists, skipping...
)

REM Check if Node.js is available
echo [4/5] Checking Node.js...
node --version >nul 2>&1
if errorlevel 1 (
    echo       WARNING: Node.js not found, skipping frontend setup
    goto :skip_frontend
)

REM Install frontend dependencies
echo [5/5] Installing frontend dependencies...
cd frontend
call npm install
cd ..
echo       Frontend dependencies installed

:skip_frontend

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo To activate the Python environment:
echo   .venv\Scripts\activate
echo.
echo To start the backend:
echo   uvicorn backend.api.main:app --reload --port 8000
echo.
echo To start the frontend:
echo   cd frontend ^&^& npm run dev
echo.
echo To start Docker services:
echo   docker-compose up -d
echo.
