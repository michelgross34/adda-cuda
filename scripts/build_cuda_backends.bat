@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Build the split CUDA kernel DLLs.  kernel.cu is the only source compiled
rem by nvcc; wrappermatvec_backend.cpp is compiled later by CMake with MinGW g++.
rem No separate Debug kernel variant is produced.
rem Usage: scripts\build_cuda_backends.bat [native|SM] [MINGW_BIN]

set "ROOT=%~dp0.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"
set "OUT=%ROOT%\cuda-backend\kernels"
set "SRC=%ROOT%\src\kernel.cu"
set "DEF=%ROOT%\src\kernel.def"
set "ARCH=%~1"
if "%ARCH%"=="" set "ARCH=native"
if /I "%ARCH:~0,3%"=="sm_" set "ARCH=%ARCH:~3%"
if /I "%ARCH%"=="native" (
  set "ARCHFLAG=-arch=native"
) else (
  set "ARCHFLAG=-arch=sm_%ARCH%"
)

if not exist "%OUT%" mkdir "%OUT%"
where nvcc.exe >nul 2>&1 || (echo ERROR: nvcc.exe not found & exit /b 1)
if not exist "%SRC%" (echo ERROR: missing %SRC% & exit /b 1)
if not exist "%DEF%" (echo ERROR: missing %DEF% & exit /b 1)

rem nvcc on Windows uses a supported MSVC host compiler for kernel.cu only.
where cl.exe >nul 2>&1
if errorlevel 1 (
  set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
  if exist "!VSWHERE!" (
    for /f "usebackq tokens=*" %%V in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSROOT=%%V"
    if defined VSROOT if exist "!VSROOT!\VC\Auxiliary\Build\vcvars64.bat" call "!VSROOT!\VC\Auxiliary\Build\vcvars64.bat" >nul
  )
)
where cl.exe >nul 2>&1 || (echo ERROR: cl.exe not found for nvcc kernel compilation & exit /b 1)

set "DLLTOOL="
if not "%~2"=="" if exist "%~2\dlltool.exe" set "DLLTOOL=%~2\dlltool.exe"
if not defined DLLTOOL for /f "delims=" %%D in ('where dlltool.exe 2^>nul') do if not defined DLLTOOL set "DLLTOOL=%%D"
if not defined DLLTOOL (
  echo ERROR: MinGW dlltool.exe not found. Pass the MinGW bin directory as argument 2.
  exit /b 1
)

call :BUILD_KERNEL double "" adda_cuda_kernels
if errorlevel 1 exit /b 1
call :BUILD_KERNEL single "-DADDA_CUDA_SINGLE_BACKEND" adda_cuda_kernels_single
if errorlevel 1 exit /b 1

echo.
echo ============================================================
echo Split CUDA kernel libraries built successfully with %ARCHFLAG%
echo   %OUT%\adda_cuda_kernels.dll
echo   %OUT%\adda_cuda_kernels_single.dll
echo The same kernel DLLs are used by CMake Debug and Release.
exit /b 0

:BUILD_KERNEL
set "LABEL=%~1"
set "PRECDEF=%~2"
set "BASE=%~3"
set "DLL=%OUT%\%BASE%.dll"
set "MSVCLIB=%OUT%\%BASE%.lib"
set "GNULIB=%OUT%\lib%BASE%.a"

echo [CUDA kernels %LABEL%] nvcc %ARCHFLAG% -O3 %PRECDEF%
del /q "%DLL%" "%MSVCLIB%" "%GNULIB%" 2>nul
nvcc -std=c++14 --shared -O3 %ARCHFLAG% %PRECDEF% ^
  -I"%ROOT%\src" "%SRC%" ^
  -Xlinker "/DEF:%DEF%" ^
  -Xlinker "/IMPLIB:%MSVCLIB%" ^
  -o "%DLL%"
if errorlevel 1 (echo ERROR: nvcc failed for %LABEL% kernels & exit /b 1)

"%DLLTOOL%" --def "%DEF%" --dllname "%BASE%.dll" --output-lib "%GNULIB%"
if errorlevel 1 (echo ERROR: dlltool failed for %LABEL% kernels & exit /b 1)
if not exist "%DLL%" (echo ERROR: missing %DLL% & exit /b 1)
if not exist "%GNULIB%" (echo ERROR: missing %GNULIB% & exit /b 1)
exit /b 0
