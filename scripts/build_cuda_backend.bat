@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ============================================================================
rem Build ONLY the CUDA backend for ADDA on Windows.
rem
rem ADDA itself is NOT compiled here.  CMake/CLion continues to use MinGW/GCC.
rem This script performs:
rem   1) nvcc + MSVC host compiler -> cuda-backend\adda_cuda_backend.dll (cuFFT + cuBLAS)
rem   2) MinGW dlltool             -> cuda-backend\libadda_cuda_backend.a
rem
rem Usage:
rem   scripts\build_cuda_backend.bat [CUDA_ARCH] [MINGW_BIN]
rem
rem Examples:
rem   scripts\build_cuda_backend.bat 86
rem   scripts\build_cuda_backend.bat 89 "C:\msys64\mingw64\bin"
rem   scripts\build_cuda_backend.bat 86 "F:\Program Files\JetBrains\CLion 2024.3.2\bin\mingw\bin"
rem
rem CUDA_ARCH is "native" (default) or the SM number without "sm_" (e.g. 75, 86, 89).
rem CUDA 11.8 supports -arch=native.
rem ============================================================================

set "ROOT=%~dp0.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"
set "OUT=%ROOT%\cuda-backend"
set "SRC=%ROOT%\src\cudamatvec_backend.cu"
set "DEF=%ROOT%\src\cudamatvec_backend.def"
set "DLL=%OUT%\adda_cuda_backend.dll"
set "MSVCLIB=%OUT%\adda_cuda_backend.lib"
set "GNULIB=%OUT%\libadda_cuda_backend.a"

set "ARCH=%~1"
if "%ARCH%"=="" set "ARCH=native"
if /I "%ARCH:~0,3%"=="sm_" set "ARCH=%ARCH:~3%"
if /I "%ARCH%"=="native" (
    set "ARCH_FLAG=-arch=native"
    set "ARCH_LABEL=native"
) else (
    set "ARCH_FLAG=-arch=sm_%ARCH%"
    set "ARCH_LABEL=sm_%ARCH%"
)

if not exist "%SRC%" (
    echo ERROR: CUDA source not found: "%SRC%"
    exit /b 1
)
if not exist "%DEF%" (
    echo ERROR: export definition file not found: "%DEF%"
    exit /b 1
)
if not exist "%OUT%" mkdir "%OUT%"

rem ---- Locate nvcc -----------------------------------------------------------
where nvcc.exe >nul 2>&1
if errorlevel 1 (
    echo ERROR: nvcc.exe not found in PATH.
    echo Open a shell where the CUDA Toolkit is available, or add CUDA\bin to PATH.
    exit /b 1
)

rem ---- Ensure a supported MSVC host compiler is active for nvcc ---------------
where cl.exe >nul 2>&1
if errorlevel 1 (
    set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    if exist "!VSWHERE!" (
        for /f "usebackq tokens=*" %%V in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSROOT=%%V"
        if defined VSROOT (
            if exist "!VSROOT!\VC\Auxiliary\Build\vcvars64.bat" (
                echo [setup] Loading MSVC x64 environment for nvcc...
                call "!VSROOT!\VC\Auxiliary\Build\vcvars64.bat" >nul
            )
        )
    )
)

where cl.exe >nul 2>&1
if errorlevel 1 (
    echo ERROR: cl.exe not found.
    echo nvcc on Windows needs a supported MSVC host compiler for the .cu file.
    echo ADDA itself will still be compiled later with MinGW/GCC.
    echo Install Visual Studio Build Tools C++ or run this script from an x64 Native Tools prompt.
    exit /b 1
)

rem ---- Locate MinGW dlltool --------------------------------------------------
set "DLLTOOL="
if not "%~2"=="" (
    if exist "%~2\dlltool.exe" set "DLLTOOL=%~2\dlltool.exe"
)
if not defined DLLTOOL (
    for /f "delims=" %%D in ('where dlltool.exe 2^>nul') do if not defined DLLTOOL set "DLLTOOL=%%D"
)
if not defined DLLTOOL (
    echo ERROR: MinGW dlltool.exe not found.
    echo Pass the MinGW bin directory as the second argument, for example:
    echo   scripts\build_cuda_backend.bat %ARCH% "C:\msys64\mingw64\bin"
    exit /b 1
)

rem ---- Remove stale outputs --------------------------------------------------
del /q "%DLL%" "%MSVCLIB%" "%GNULIB%" 2>nul

rem ---- 1/2: build CUDA DLL using nvcc only ----------------------------------
echo.
echo [CUDA 1/2] nvcc -^> adda_cuda_backend.dll  ^(%ARCH_LABEL%^)
echo.

nvcc -std=c++14 --shared -O3 %ARCH_FLAG% ^
    -I"%ROOT%\src" ^
    "%SRC%" ^
    -lcufft ^
    -lcublas ^
    -Xlinker "/DEF:%DEF%" ^
    -Xlinker "/IMPLIB:%MSVCLIB%" ^
    -o "%DLL%"

if errorlevel 1 (
    echo.
    echo ERROR: nvcc compilation/link failed.
    exit /b 1
)
if not exist "%DLL%" (
    echo ERROR: nvcc returned success but DLL was not created: "%DLL%"
    exit /b 1
)

rem ---- 2/2: create GNU import library for MinGW ------------------------------
echo.
echo [CUDA 2/2] dlltool -^> libadda_cuda_backend.a
"%DLLTOOL%" --def "%DEF%" --dllname "adda_cuda_backend.dll" --output-lib "%GNULIB%"
if errorlevel 1 (
    echo.
    echo ERROR: dlltool failed.
    exit /b 1
)
if not exist "%GNULIB%" (
    echo ERROR: GNU import library was not created: "%GNULIB%"
    exit /b 1
)

echo.
echo ================================================================
echo CUDA backend built successfully.
echo   DLL       : %DLL%
echo   MSVC lib  : %MSVCLIB%
echo   MinGW lib : %GNULIB%
echo ================================================================
echo.
echo Next, build ADDA with MinGW/CLion using:
echo   -DADDA_CUDA=ON

echo CMake must NOT compile cudamatvec_backend.cu.
exit /b 0
