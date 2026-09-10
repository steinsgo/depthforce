@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo DepthForce virtual environment was not found.
    echo Expected: %~dp0.venv\Scripts\python.exe
    echo.
    pause
    exit /b 1
)

title DepthForce Launcher
echo Starting DepthForce...
echo Keep the camera view empty during startup, or step away and press R to relearn it.
echo Keys: 1 = Balanced, 2 = Punchy, R = relearn background, Q = quit.
echo.

".venv\Scripts\python.exe" main.py --preset balanced --debug --camera-debug %*
set "depthforce_exit=%errorlevel%"

if not "%depthforce_exit%"=="0" (
    echo.
    echo DepthForce exited with error code %depthforce_exit%.
    pause
)

exit /b %depthforce_exit%
