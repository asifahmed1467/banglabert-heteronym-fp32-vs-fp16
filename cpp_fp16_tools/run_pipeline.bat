@echo off
rem Full conversion in C++:  self-test -> fp32 folder -> fp16 folder -> proof of every value.
rem Usage:  run_pipeline.bat C:\Users\Asif\heteronym_models_fp32 C:\Users\Asif\heteronym_models_fp16_cpp
setlocal
cd /d "%~dp0"
if "%~2"=="" (
  echo usage: run_pipeline.bat ^<fp32 folder^> ^<new fp16 folder^>
  exit /b 2
)
set EXE=build\Release
if not exist %EXE%\fp16_convert.exe (
  echo Build first: build.bat
  exit /b 1
)
echo === 0/3 self-test of the rounding code ===
%EXE%\fp16_selftest.exe
if errorlevel 1 exit /b 1
echo.
echo === 1/3 convert fp32 to fp16 ===
%EXE%\fp16_convert.exe --src %1 --out %2
if errorlevel 1 exit /b 1
echo.
echo === 2/3 verify every value ===
%EXE%\fp16_verify.exe --fp32 %1 --fp16 %2 --full
exit /b %errorlevel%
