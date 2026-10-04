@echo off
REM Run once as Administrator: lets other PCs on the local network open the app
netsh advfirewall firewall add rule name="TarhYab 8501" dir=in action=allow protocol=TCP localport=8501 profile=private
pause
