@echo off
rem One-click launcher for the posture monitor.
rem
rem Double-click this file. On the first run it builds the virtual environment
rem and downloads the pose model, which takes a few minutes; after that it goes
rem straight to starting the cameras and opening the control panel.
rem
rem Anything "python -m posture" accepts can be passed here too, for example:
rem   start-posture.bat --list-cameras
rem   start-posture.bat --diagnose
setlocal
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"

if exist "%PY%" goto deps

echo Creating the virtual environment (first run only)...
py -3 -m venv .venv >nul 2>&1
if not exist "%PY%" python -m venv .venv
if not exist "%PY%" goto nopython

:deps
rem Cheaper than asking pip: importing what the app actually needs also
rem catches a half-finished install from an interrupted first run.
"%PY%" -c "import mediapipe, cv2, numpy, pystray, PIL" >nul 2>&1
if not errorlevel 1 goto model
echo Installing dependencies (first run only, this takes a few minutes)...
"%PY%" -m pip install --upgrade pip >nul 2>&1
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 goto pipfailed

:model
if exist "models\pose_landmarker_lite.task" goto run
echo Downloading the pose model (first run only, 5.6 MB)...
"%PY%" scripts\fetch_model.py
if errorlevel 1 goto modelfailed

:run
echo.
"%PY%" -m posture %*
set "CODE=%ERRORLEVEL%"
rem Ctrl-C reaches the app first and it shuts the cameras down cleanly; cmd
rem then asks "Terminate batch job (Y/N)?" -- either answer is fine by then.
if not "%CODE%"=="0" goto appfailed
exit /b 0

:nopython
echo.
echo Could not find Python. Install Python 3.11 or newer from python.org,
echo ticking "Add python.exe to PATH", then run this file again.
goto stop

:pipfailed
echo.
echo Installing the dependencies failed; the lines above say why.
goto stop

:modelfailed
echo.
echo Downloading the pose model failed -- check the network connection.
echo This is the only step that needs the internet: the app itself has no
echo networking code and never reaches out.
goto stop

:appfailed
echo.
echo Posture exited with code %CODE%.
echo If it says the control panel port is in use, a copy is probably already
echo running -- open http://127.0.0.1:8760/ instead of starting a second one.
goto stop

:stop
echo.
pause
exit /b 1
