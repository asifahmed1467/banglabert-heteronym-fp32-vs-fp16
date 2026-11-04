@echo off
rem Builds the C++ tools. Usage:  build.bat [path to unpacked onnxruntime folder]
rem Run it from "x64 Native Tools Command Prompt for VS" (or any prompt where cmake and cl work).
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  cmake -S . -B build
) else (
  cmake -S . -B build -DONNXRUNTIME_DIR=%~1
)
if errorlevel 1 exit /b 1
cmake --build build --config Release
if errorlevel 1 exit /b 1
echo.
echo Built into build\Release\
dir /b build\Release\*.exe
