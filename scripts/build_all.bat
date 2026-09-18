@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Standalone CMake build: no CLion project or CLion-provided tool is needed.
rem Build every Windows executable and collect runtime files in build_windows\bin.
rem Parallelism is detected per machine; ADDA_BUILD_JOBS can override it.
rem Usage: scripts\build_all.bat [CUDA_ARCH] [MINGW_BIN]
rem Example: scripts\build_all.bat 89 "C:\msys64\mingw64\bin"

set "ROOT=%~dp0.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"
set "BUILD=%ROOT%\build_windows"
set "BIN=%BUILD%\bin"
set "BACKEND=%ROOT%\cuda-backend"
set "ARCH=%~1"
set "MINGW_BIN=%~2"

if "%ARCH%"=="" set "ARCH=native"

where cmake.exe >nul 2>&1
if errorlevel 1 (
    echo ERROR: cmake.exe not found in PATH.
    exit /b 1
)

set "GCC="
if defined MINGW_BIN (
    if not exist "%MINGW_BIN%\gcc.exe" (
        echo ERROR: gcc.exe not found in "%MINGW_BIN%".
        exit /b 1
    )
    set "GCC=%MINGW_BIN%\gcc.exe"
    set "GXX=%MINGW_BIN%\g++.exe"
) else (
    for /f "delims=" %%G in ('where gcc.exe 2^>nul') do if not defined GCC set "GCC=%%G"
    if not defined GCC (
        echo ERROR: gcc.exe not found in PATH.
        exit /b 1
    )
    for %%G in ("!GCC!") do set "MINGW_BIN=%%~dpG"
    set "GXX=!MINGW_BIN!g++.exe"
)

rem Put the selected MinGW toolchain first so CMake can discover all GNU
rem compilers, including GFortran, when Fortran is enabled in CMakeLists.txt.
if not exist "%MINGW_BIN%\g++.exe" (echo ERROR: g++.exe not found in "%MINGW_BIN%" & exit /b 1)
set "GXX=%MINGW_BIN%\g++.exe"

set "PATH=%MINGW_BIN%;%PATH%"

set "MINGW_MAKE="
if exist "%MINGW_BIN%\mingw32-make.exe" set "MINGW_MAKE=%MINGW_BIN%\mingw32-make.exe"
if not defined MINGW_MAKE (
    for /f "delims=" %%M in ('where mingw32-make.exe 2^>nul') do if not defined MINGW_MAKE set "MINGW_MAKE=%%M"
)
set "NINJA="
for /f "delims=" %%N in ('where ninja.exe 2^>nul') do if not defined NINJA set "NINJA=%%N"

set "CACHED_GENERATOR="
if exist "%BUILD%\CMakeCache.txt" (
    for /f "tokens=2 delims==" %%C in ('findstr /b /c:"CMAKE_GENERATOR:INTERNAL=" "%BUILD%\CMakeCache.txt"') do set "CACHED_GENERATOR=%%C"
)
set "GENERATOR="
set "MAKE_PROGRAM="
if /I "!CACHED_GENERATOR!"=="MinGW Makefiles" (
    set "GENERATOR=MinGW Makefiles"
    set "MAKE_PROGRAM=!MINGW_MAKE!"
)
if /I "!CACHED_GENERATOR!"=="Ninja" (
    set "GENERATOR=Ninja"
    set "MAKE_PROGRAM=!NINJA!"
)
if not defined CACHED_GENERATOR (
    if defined MINGW_MAKE (
        set "GENERATOR=MinGW Makefiles"
        set "MAKE_PROGRAM=!MINGW_MAKE!"
    ) else if defined NINJA (
        set "GENERATOR=Ninja"
        set "MAKE_PROGRAM=!NINJA!"
    )
)
if not defined GENERATOR (
    if defined CACHED_GENERATOR (
        echo ERROR: unsupported cached CMake generator: "!CACHED_GENERATOR!".
        echo Remove "%BUILD%" or configure it with Ninja or MinGW Makefiles.
    ) else (
        echo ERROR: neither mingw32-make.exe nor ninja.exe was found.
        echo Install a MinGW make tool or Ninja; CLion is not required.
    )
    exit /b 1
)
if not defined MAKE_PROGRAM (
    echo ERROR: build tool missing for CMake generator "!GENERATOR!".
    exit /b 1
)

set "JOBS=%ADDA_BUILD_JOBS%"
if not defined JOBS set "JOBS=%NUMBER_OF_PROCESSORS%"
if not defined JOBS set "JOBS=1"
for /f "delims=0123456789" %%J in ("%JOBS%") do set "JOBS=1"
if %JOBS% LSS 1 set "JOBS=1"

echo [setup] CMake generator: !GENERATOR!
echo [setup] Build tool: !MAKE_PROGRAM!
echo [setup] Parallel jobs: %JOBS%

if not exist "%ROOT%\fftw\fftw3.h" (
    echo ERROR: FFTW header not found: "%ROOT%\fftw\fftw3.h"
    exit /b 1
)
if not exist "%ROOT%\fftw\libfftw3.dll.a" (
    echo ERROR: FFTW double import library not found: "%ROOT%\fftw\libfftw3.dll.a"
    exit /b 1
)
if not exist "%ROOT%\fftw\libfftw3f.dll.a" (
    echo ERROR: FFTW float32 import library not found: "%ROOT%\fftw\libfftw3f.dll.a"
    exit /b 1
)

echo [1/4] Building CUDA backends
call "%~dp0build_cuda_backends.bat" "%ARCH%" "%MINGW_BIN%"
if errorlevel 1 exit /b 1

rem build_cuda_backends.bat always builds both CUDA implementations.  This
rem build_all currently uses split mode, but verify the monolithic libraries as
rem well so they are ready for ADDA_CUDA_SPLIT_BACKEND=OFF / CLion profiles.
for %%F in (
    "%BACKEND%\kernels\adda_cuda_kernels.dll"
    "%BACKEND%\kernels\libadda_cuda_kernels.a"
    "%BACKEND%\kernels\adda_cuda_kernels_single.dll"
    "%BACKEND%\kernels\libadda_cuda_kernels_single.a"
    "%BACKEND%\release\adda_cuda_backend.dll"
    "%BACKEND%\release\libadda_cuda_backend.a"
    "%BACKEND%\release\adda_cuda_backend_single.dll"
    "%BACKEND%\release\libadda_cuda_backend_single.a"
) do (
    if not exist "%%~F" (
        echo ERROR: CUDA build output missing: "%%~F"
        exit /b 1
    )
)

echo [2/4] Configuring CMake in "%BUILD%"
cmake -S "%ROOT%" -B "%BUILD%" -G "!GENERATOR!" ^
    "-DCMAKE_MAKE_PROGRAM=!MAKE_PROGRAM!" ^
    -DCMAKE_BUILD_TYPE=Release ^
    "-DCMAKE_C_COMPILER=%GCC%" ^
    "-DCMAKE_CXX_COMPILER=%GXX%" ^
    -DADDA_CUDA_SPLIT_BACKEND=ON ^
    -DADDA_NO_GITHASH=ON ^
    -DADDA_SINGLE=ON ^
    -DADDA_CUDA=ON ^
    -DADDA_CUDA_SLICE=ON ^
    -DADDA_CUDA_LOW_MEM=ON ^
    -DADDA_CUDA_SINGLE=ON ^
    -DADDA_CUDA_SINGLE_SLICE=ON ^
    -DADDA_CUDA_LOW_MEM_SINGLE=ON ^
    "-DFFTW3_INCLUDE_DIR=%ROOT%\fftw" ^
    "-DFFTW3_LIBRARY=%ROOT%\fftw\libfftw3.dll.a" ^
    "-DFFTW3F_LIBRARY=%ROOT%\fftw\libfftw3f.dll.a"
if errorlevel 1 exit /b 1

echo [3/4] Building all eight executables
cmake --build "%BUILD%" --config Release --parallel %JOBS% --target ^
    adda adda_single adda_cuda adda_cuda_slice adda_low_mem_double ^
    adda_cuda_single adda_single_slice adda_low_mem
if errorlevel 1 exit /b 1

if not exist "%BIN%" mkdir "%BIN%"

echo [4/4] Collecting DLLs and AI resources
call :COPY_REQUIRED "%BACKEND%\kernels\adda_cuda_kernels.dll"
if errorlevel 1 exit /b 1
call :COPY_REQUIRED "%BACKEND%\kernels\adda_cuda_kernels_single.dll"
if errorlevel 1 exit /b 1
call :COPY_REQUIRED "%ROOT%\fftw\libfftw3-3.dll"
if errorlevel 1 exit /b 1
call :COPY_REQUIRED "%ROOT%\fftw\libfftw3f-3.dll"
if errorlevel 1 exit /b 1
call :COPY_OPTIONAL "%MINGW_BIN%\libgcc_s_seh-1.dll"
call :COPY_OPTIONAL "%MINGW_BIN%\libgcc_s_dw2-1.dll"
call :COPY_OPTIONAL "%MINGW_BIN%\libwinpthread-1.dll"
for %%D in ("%MINGW_BIN%\libgfortran-*.dll") do call :COPY_OPTIONAL "%%~fD"
for %%D in ("%MINGW_BIN%\libquadmath-*.dll") do call :COPY_OPTIONAL "%%~fD"
for %%D in ("%MINGW_BIN%\libssp-*.dll") do call :COPY_OPTIONAL "%%~fD"

if not exist "%ROOT%\ai\lanier_tqc_v1_ema_actor.bin" (
    echo ERROR: AI resource not found: "%ROOT%\ai\lanier_tqc_v1_ema_actor.bin"
    exit /b 1
)
if not exist "%BIN%\ai" mkdir "%BIN%\ai"
xcopy "%ROOT%\ai\*" "%BIN%\ai\" /E /I /Y >nul
if errorlevel 1 exit /b 1

set "MISSING="
for %%E in (
    adda.exe
    adda_single.exe
    adda_cuda.exe
    adda_cuda_slice.exe
    adda_low_mem_double.exe
    adda_cuda_single.exe
    adda_single_slice.exe
    adda_low_mem.exe
) do if not exist "%BIN%\%%E" set "MISSING=!MISSING! %%E"
if defined MISSING (
    echo ERROR: missing executables in "%BIN%":!MISSING!
    exit /b 1
)

echo.
echo ================================================================
echo Complete build available in: %BIN%
echo   8 executables
echo   CUDA, FFTW and MinGW runtime DLLs
echo   AI resources: %BIN%\ai
echo ================================================================
exit /b 0

:COPY_REQUIRED
if not exist "%~1" (
    echo ERROR: required DLL not found: "%~1"
    exit /b 1
)
copy /Y "%~1" "%BIN%\" >nul
if errorlevel 1 exit /b 1
exit /b 0

:COPY_OPTIONAL
if exist "%~1" copy /Y "%~1" "%BIN%\" >nul
exit /b 0
