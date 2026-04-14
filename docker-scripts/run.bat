@echo off
REM Run script for Nepal Real Estate Docker container (Windows)

setlocal enabledelayedexpansion

REM Configuration
set IMAGE_NAME=nepal-realestate
set TAG=latest
set CONTAINER_NAME=nepal-realestate-app
set PORT=8501

echo.
echo ========================================
echo   Running Nepal Real Estate Container
echo ========================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running. Please start Docker Desktop and try again.
    pause
    exit /b 1
)

echo [SUCCESS] Docker is running
echo.

REM Check if image exists
docker images %IMAGE_NAME%:%TAG% --format "{{.Repository}}" | findstr %IMAGE_NAME% >nul
if errorlevel 1 (
    echo [ERROR] Docker image '%IMAGE_NAME%:%TAG%' not found!
    echo.
    echo Please build the image first:
    echo   docker-scripts\build.bat
    echo   OR
    echo   docker build -t %IMAGE_NAME%:%TAG% .
    pause
    exit /b 1
)

echo [SUCCESS] Docker image found
echo.

REM Stop existing container if running
docker ps -q -f name=%CONTAINER_NAME% >nul 2>&1
if not errorlevel 1 (
    echo [WARNING] Stopping existing container...
    docker stop %CONTAINER_NAME% >nul
)

REM Remove existing container if exists
docker ps -aq -f name=%CONTAINER_NAME% >nul 2>&1
if not errorlevel 1 (
    echo [WARNING] Removing existing container...
    docker rm %CONTAINER_NAME% >nul
)

REM Check if .env file exists
if exist ".env" (
    echo [SUCCESS] .env file found - will use environment variables
    set ENV_FILE_OPTION=--env-file .env
) else (
    echo [WARNING] .env file not found
    echo [INFO] Container will run without API tokens ^(RAG chatbot won't work^)
    set ENV_FILE_OPTION=
)

REM Check if port is available
netstat -an | findstr ":%PORT% " >nul
if not errorlevel 1 (
    echo [ERROR] Port %PORT% is already in use!
    echo.
    echo Please stop the service using port %PORT% or use a different port:
    echo   docker run -p 8502:8501 %ENV_FILE_OPTION% %IMAGE_NAME%:%TAG%
    pause
    exit /b 1
)

echo [SUCCESS] Port %PORT% is available
echo.

REM Run container
echo [INFO] Starting container...
echo Command: docker run -d -p %PORT%:8501 --name %CONTAINER_NAME% %ENV_FILE_OPTION% --restart unless-stopped %IMAGE_NAME%:%TAG%
echo.

docker run -d -p %PORT%:8501 --name %CONTAINER_NAME% %ENV_FILE_OPTION% --restart unless-stopped %IMAGE_NAME%:%TAG%

if errorlevel 1 (
    echo [ERROR] Failed to start container!
    pause
    exit /b 1
)

echo [SUCCESS] Container started successfully!
echo.

REM Wait a moment for container to start
timeout /t 3 /nobreak >nul

REM Check if container is running
docker ps -q -f name=%CONTAINER_NAME% >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Container failed to start!
    echo [INFO] Checking logs...
    docker logs %CONTAINER_NAME%
    pause
    exit /b 1
)

echo [SUCCESS] Container is running
echo.

REM Show container info
echo [INFO] Container details:
docker ps -f name=%CONTAINER_NAME% --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo.

echo ========================================
echo   Your app is now running! 🎉
echo ========================================
echo.
echo 📱 Access your app:
echo    Local:    http://localhost:%PORT%
echo    Network:  http://YOUR_IP:%PORT%
echo.
echo 📊 Useful commands:
echo    View logs:     docker logs -f %CONTAINER_NAME%
echo    Stop app:      docker stop %CONTAINER_NAME%
echo    Start app:     docker start %CONTAINER_NAME%
echo    Restart app:   docker restart %CONTAINER_NAME%
echo    Remove app:    docker rm -f %CONTAINER_NAME%
echo.

REM Show recent logs
echo [INFO] Recent logs ^(last 10 lines^):
docker logs --tail 10 %CONTAINER_NAME%
echo.

echo 💡 To follow logs in real-time:
echo    docker logs -f %CONTAINER_NAME%
echo.

echo ========================================
echo   Container is running successfully! 🎉
echo ========================================
echo.
echo Press any key to exit...
pause >nul