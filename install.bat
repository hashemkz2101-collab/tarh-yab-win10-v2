@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0"
echo ===== Tarh-Yab install =====

rem ---- choose a Python version (3.12 is the safest) ----
set PYV=
for %%v in (3.12 3.11 3.10 3.13) do (
  if not defined PYV (
    py -%%v --version >nul 2>&1 && set PYV=-%%v
  )
)
if not defined PYV set PYV=-3
py %PYV% --version
if errorlevel 1 (
  echo Python was not found. Install Python 3.12 from python.org and tick "Add python.exe to PATH".
  pause
  exit /b 1
)

rem ---- virtual environment + packages (skipped if already done) ----
if not exist venv\pip_ok.txt (
  if exist venv rmdir /s /q venv
  py %PYV% -m venv venv
  if errorlevel 1 (
    echo Could not create the virtual environment.
    pause
    exit /b 1
  )
  call venv\Scripts\activate.bat
  python -m pip install --upgrade pip >nul 2>&1
  set OK=0
  for %%M in ("default" "-i https://mirror-pypi.runflare.com/simple" "-i https://pypi.jamko.ir/simple") do (
    if "!OK!"=="0" (
      set "ARGS=%%~M"
      if "!ARGS!"=="default" set "ARGS="
      echo.
      echo --- pip install !ARGS! ---
      pip install -r requirements.txt !ARGS! && set OK=1
    )
  )
  if "!OK!"=="0" (
    echo.
    echo *** Package install FAILED. Check your internet connection / VPN / DNS and run install.bat again. ***
    pause
    exit /b 1
  )
  echo ok> venv\pip_ok.txt
) else (
  call venv\Scripts\activate.bat
)

rem ---- models ----
python download_model.py
if errorlevel 1 (
  echo Retrying the model download through hf-mirror.com ...
  set HF_ENDPOINT=https://hf-mirror.com
  python download_model.py
)
if not exist model_dinov2_small\model.safetensors (
  echo.
  echo *** Model download FAILED. ***
  echo Download on a PC with internet: run download_model.py there, then copy the folder
  echo model_dinov2_small next to app.py here, and run install.bat again.
  pause
  exit /b 1
)

echo.
echo Install finished successfully. Put sheet images in the "library" folder, then run run.bat
pause
