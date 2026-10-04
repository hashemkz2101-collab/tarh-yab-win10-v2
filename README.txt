Tarh-Yab v2.1 for Windows 10/11
1) Install Python 3.10-3.12 (64-bit) from python.org, tick "Add python.exe to PATH".
2) Double-click install.bat (internet needed once; downloads two models, about 450 MB).
   If pip cannot reach pypi.org, the script retries with Iranian mirrors automatically.
   Python 3.12 is recommended (3.14 may lack some packages).
3) Copy sheet images into "library" (or add them from the sidebar inside the app).
4) Double-click run.bat -> open http://localhost:8501
   First run builds the index; later runs only read new sheets.
Optional: run allow-network.bat as Administrator, then open http://<this-PC-IP>:8501 from another PC.

Upgrading from v1: copy all files over the old folder (keep "library" and "venv"), then run install.bat again.

Measure accuracy on your own samples (collage images: customer photo on the left, library sheet on the right):
   venv\Scripts\activate  then  python evaluate.py C:\path\to\collages
