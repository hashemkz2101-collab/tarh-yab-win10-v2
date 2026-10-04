@echo off
cd /d "%~dp0"
if not exist venv\pip_ok.txt (
  echo Installation is not complete. Run install.bat first.
  pause
  exit /b 1
)
call venv\Scripts\activate.bat
echo Your IP addresses (use one of them from another PC, e.g. http://IP:8501):
ipconfig | findstr /i "IPv4"
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
pause
