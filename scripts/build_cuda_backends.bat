@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Build BOTH CUDA 11.8 backends separately from the MinGW/CMake build.
rem   double : adda_cuda_backend.dll
rem   float32: adda_cuda_backend_single.dll (ADDA_CUDA_SINGLE_BACKEND)
rem Large float32 vectors/FFT stay FP32; scalar reductions use FP64 cuBLAS.
rem
rem Usage:
rem   scripts\build_cuda_backends.bat [native|SM] [MINGW_BIN]
rem Examples:
rem   scripts\build_cuda_backends.bat
rem   scripts\build_cuda_backends.bat native
rem   scripts\build_cuda_backends.bat 86 "C:\msys64\mingw64\bin"

set "ROOT=%~dp0.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"
set "OUT=%ROOT%\cuda-backend"
set "SRC=%ROOT%\src\cudamatvec_backend.cu"
set "DEF=%ROOT%\src\cudamatvec_backend.def"
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

rem nvcc 11.8 on Windows needs a supported host C++ compiler (normally MSVC cl.exe).
where cl.exe >nul 2>&1
if errorlevel 1 (
  set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
  if exist "!VSWHERE!" (
    for /f "usebackq tokens=*" %%V in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSROOT=%%V"
    if defined VSROOT if exist "!VSROOT!\VC\Auxiliary\Build\vcvars64.bat" call "!VSROOT!\VC\Auxiliary\Build\vcvars64.bat" >nul
  )
)
where cl.exe >nul 2>&1 || (echo ERROR: cl.exe not found for nvcc host compilation & exit /b 1)

set "DLLTOOL="
if not "%~2"=="" if exist "%~2\dlltool.exe" set "DLLTOOL=%~2\dlltool.exe"
if not defined DLLTOOL for /f "delims=" %%D in ('where dlltool.exe 2^>nul') do if not defined DLLTOOL set "DLLTOOL=%%D"
if not defined DLLTOOL (
  echo ERROR: MinGW dlltool.exe not found.
  echo Pass the MinGW bin directory as argument 2.
  exit /b 1
)

call :BUILD_BACKEND double "" "adda_cuda_backend"
if errorlevel 1 exit /b 1
call :BUILD_BACKEND single "-DADDA_CUDA_SINGLE_BACKEND" "adda_cuda_backend_single"
if errorlevel 1 exit /b 1

echo.
echo ============================================================
echo Both CUDA backends built successfully with %ARCHFLAG%
echo   %OUT%\adda_cuda_backend.dll
 echo  %OUT%\adda_cuda_backend_single.dll
 echo  %OUT%\libadda_cuda_backend.a
 echo  %OUT%\libadda_cuda_backend_single.a
 echo.
echo Rebuild the CLion target afterwards so the correct DLL is copied to bin.
exit /b 0

:BUILD_BACKEND
set "LABEL=%~1"
set "PRECDEF=%~2"
set "BASE=%~3"
set "DLL=%OUT%\%BASE%.dll"
set "MSVCLIB=%OUT%\%BASE%.lib"
set "GNULIB=%OUT%\lib%BASE%.a"

echo.
echo [CUDA %LABEL%] nvcc %ARCHFLAG% %PRECDEF%
del /q "%DLL%" "%MSVCLIB%" "%GNULIB%" 2>nul
nvcc -std=c++14 --shared -O3 %ARCHFLAG% %PRECDEF% ^
  -I"%ROOT%\src" ^
  "%SRC%" ^
  -lcufft -lcublas ^
  -Xlinker "/DEF:%DEF%" ^
  -Xlinker "/IMPLIB:%MSVCLIB%" ^
  -o "%DLL%"
if errorlevel 1 (echo ERROR: nvcc failed for %LABEL% backend & exit /b 1)

"%DLLTOOL%" --def "%DEF%" --dllname "%BASE%.dll" --output-lib "%GNULIB%"
if errorlevel 1 (echo ERROR: dlltool failed for %LABEL% backend & exit /b 1)
if not exist "%DLL%" (echo ERROR: missing %DLL% & exit /b 1)
if not exist "%GNULIB%" (echo ERROR: missing %GNULIB% & exit /b 1)
exit /b 0
