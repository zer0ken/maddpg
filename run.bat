pushd "%~dp0"
cd %~dp0
call %~dp0venv\Scripts\activate.bat
pip3 install -r requirements.txt
python main.py
pause
