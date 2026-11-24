@echo off
REM Start the ASL Fingerspelling app on Windows (after following README setup).
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
  echo Virtual environment not found. Follow the Setup steps in README.md first.
  pause
  exit /b 1
)
if not exist frontend\dist\index.html (
  echo Building the browser interface ^(first run only^)...
  call npm --prefix frontend install || goto :error
  call npm --prefix frontend run build || goto :error
)
start "" http://127.0.0.1:8000
venv\Scripts\python.exe -m backend.server
goto :eof
:error
echo Build failed. Is Node.js installed? See README.md.
pause
