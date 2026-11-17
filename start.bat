@echo off
setlocal
title RAG Q&A

echo.
echo  ================================
echo   RAG Q^&A  --  Starting up...
echo  ================================
echo.

:: ── Activate virtual environment ─────────────────────────────────────────────
if not exist ".venv\Scripts\activate.bat" (
    echo  [ERROR] Virtual environment not found.
    echo          Run:  python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
    pause & exit /b 1
)
call .venv\Scripts\activate.bat

:: ── Build index if it doesn't exist yet ──────────────────────────────────────
if not exist "index\index.faiss" (
    echo  [INFO] No index found -- building it now from PDFs in data\...
    echo.
    python ingest.py
    if errorlevel 1 (
        echo.
        echo  [ERROR] Indexing failed. Put at least one PDF in the data\ folder and try again.
        pause & exit /b 1
    )
    echo.
)

:: ── Open browser after a short delay (runs in background) ────────────────────
start "" cmd /c "timeout /t 2 >nul && start http://127.0.0.1:8000"

:: ── Start the server (stays open until you press Ctrl+C) ─────────────────────
echo  [INFO] Server starting at http://127.0.0.1:8000
echo  [INFO] Press Ctrl+C to stop.
echo.
uvicorn app:app --host 127.0.0.1 --port 8000

echo.
echo  [INFO] Server stopped.
pause
