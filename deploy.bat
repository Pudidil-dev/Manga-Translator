@echo off
REM Manga Translator - Quick Deploy Script for Windows
REM Usage: deploy.bat [option]
REM   Options: install, run, docker, stop

setlocal enabledelayedexpansion

cd /d "%~dp0"

REM ==============================================================================
REM Parse command
REM ==============================================================================
if "%1"=="" goto :help
if "%1"=="install" goto :install
if "%1"=="dev" goto :dev
if "%1"=="run" goto :dev
if "%1"=="prod" goto :prod
if "%1"=="docker" goto :docker
if "%1"=="compose" goto :compose
if "%1"=="stop" goto :stop
if "%1"=="help" goto :help
if "%1"=="--help" goto :help
if "%1"=="-h" goto :help

echo Unknown command: %1
goto :help

REM ==============================================================================
REM Install dependencies
REM ==============================================================================
:install
echo ==================================
echo Installing Manga Translator
echo ==================================

REM Check Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10+
    exit /b 1
)

python --version
echo.

REM Create virtual environment if not exists
if not exist ".venv" (
    echo [*] Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install dependencies
echo [*] Installing dependencies...
pip install -r requirements.txt

REM Check models
echo [*] Checking models...
python download_models.py status

echo.
echo [OK] Installation complete!
echo.
echo To activate the environment:
echo   .venv\Scripts\activate
echo.
echo To run the server:
echo   python app.py
goto :end

REM ==============================================================================
REM Run development server
REM ==============================================================================
:dev
echo ==================================
echo Starting Development Server
echo ==================================

REM Activate virtual environment
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM Check if model exists
if not exist "model\model.pt" (
    echo [WARNING] Model not found. Please ensure model\model.pt exists.
)

REM Run Flask app
echo [*] Starting Flask server on http://localhost:5000
python app.py
goto :end

REM ==============================================================================
REM Run production server
REM ==============================================================================
:prod
echo ==================================
echo Starting Production Server
echo ==================================

REM Activate virtual environment
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM Check waitress (Windows WSGI server)
pip show waitress >nul 2>nul
if %errorlevel% neq 0 (
    echo [*] Installing waitress...
    pip install waitress
)

REM Set environment variables
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4

REM Preload models
echo [*] Preloading models...
python -c "from services import Services; Services.preload_all()" 2>nul

REM Run with waitress (Windows-compatible)
echo [*] Starting Waitress server on http://0.0.0.0:5000
python -c "from waitress import serve; from app import app; serve(app, host='0.0.0.0', port=5000, threads=4)"
goto :end

REM ==============================================================================
REM Docker build and run
REM ==============================================================================
:docker
echo ==================================
echo Docker Deployment
echo ==================================

REM Check Docker
where docker >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Docker not found. Please install Docker Desktop.
    exit /b 1
)

REM Build image
echo [*] Building Docker image...
docker build -t manga-translator .

REM Stop existing container
docker stop manga-translator 2>nul
docker rm manga-translator 2>nul

REM Run container
echo [*] Starting container...
docker run -d ^
    --name manga-translator ^
    -p 5000:5000 ^
    -v "%cd%\model:/app/model" ^
    -v "%cd%\fonts:/app/fonts" ^
    --restart unless-stopped ^
    manga-translator

echo.
echo [OK] Container started!
echo.
echo API available at: http://localhost:5000
echo.
echo View logs: docker logs -f manga-translator
echo Stop: docker stop manga-translator
goto :end

REM ==============================================================================
REM Docker Compose
REM ==============================================================================
:compose
echo ==================================
echo Docker Compose Deployment
echo ==================================

where docker-compose >nul 2>nul
if %errorlevel% neq 0 (
    docker compose version >nul 2>nul
    if %errorlevel% neq 0 (
        echo [ERROR] Docker Compose not found.
        exit /b 1
    )
    set COMPOSE_CMD=docker compose
) else (
    set COMPOSE_CMD=docker-compose
)

echo [*] Starting services...
%COMPOSE_CMD% up -d --build

echo.
echo [OK] Services started!
echo.
echo API available at: http://localhost:5000
echo.
echo View logs: %COMPOSE_CMD% logs -f
echo Stop: %COMPOSE_CMD% down
goto :end

REM ==============================================================================
REM Stop all services
REM ==============================================================================
:stop
echo ==================================
echo Stopping All Services
echo ==================================

REM Stop Docker container
docker stop manga-translator 2>nul && echo [OK] Stopped Docker container

REM Stop Docker Compose
docker-compose down 2>nul && echo [OK] Stopped Docker Compose
docker compose down 2>nul

REM Kill Python processes (be careful with this)
echo [*] Note: To stop Flask/Waitress, use Ctrl+C in the running terminal
echo [OK] Services stopped
goto :end

REM ==============================================================================
REM Show help
REM ==============================================================================
:help
echo Manga Translator - Deploy Script (Windows)
echo.
echo Usage: deploy.bat [command]
echo.
echo Commands:
echo   install     Install dependencies and setup environment
echo   dev         Run development server (Flask)
echo   prod        Run production server (Waitress)
echo   docker      Build and run Docker container
echo   compose     Run with Docker Compose
echo   stop        Stop Docker services
echo   help        Show this help message
echo.
echo Examples:
echo   deploy.bat install    # First time setup
echo   deploy.bat dev        # Development
echo   deploy.bat prod       # Production
echo   deploy.bat docker     # Docker deployment
goto :end

:end
endlocal
