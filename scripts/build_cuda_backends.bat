@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Build BOTH CUDA backend variants, independently of the CMake mode:
rem   1) split kernels: kernel.cu -> cuda-backend\kernels\adda_cuda_kernels*.dll
rem   2) monolithic backend: cudamatvec_backend.cu -> cuda-backend\release\adda_cuda_backend*.dll
rem
rem CMake later decides which variant is used through ADDA_CUDA_SPLIT_BACKEND.
rem In split mode wrappermatvec_backend.cpp is compiled later by CMake with MinGW g++.
rem Usage: scripts\build_cuda_backends.bat [native|SM] [MINGW_BIN]

set "ROOT=%~dp0.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"

set "KERNEL_OUT=%ROOT%\cuda-backend\kernels"
set "BACKEND_OUT=%ROOT%\cuda-backend\release"

set "KERNEL_SRC=%ROOT%\src\kernel.cu"
set "KERNEL_DEF=%ROOT%\src\kernel.def"
set "BACKEND_SRC=%ROOT%\src\cudamatvec_backend.cu"
set "BACKEND_DEF=%ROOT%\src\cudamatvec_backend.def"
set "BACKEND_SINGLE_DEF=%ROOT%\src\cudamatvec_backend_single.def"

set "ARCH=%~1"
if "%ARCH%"=="" set "ARCH=native"
if /I "%ARCH:~0,3%"=="sm_" set "ARCH=%ARCH:~3%"
if /I "%ARCH%"=="native" (
  set "ARCHFLAG=-arch=native"
) else (
  set "ARCHFLAG=-arch=sm_%ARCH%"
)

if not exist "%KERNEL_OUT%" mkdir "%KERNEL_OUT%"
if not exist "%BACKEND_OUT%" mkdir "%BACKEND_OUT%"

where nvcc.exe >nul 2>&1 || (echo ERROR: nvcc.exe not found & exit /b 1)
if not exist "%KERNEL_SRC%" (echo ERROR: missing %KERNEL_SRC% & exit /b 1)
if not exist "%KERNEL_DEF%" (echo ERROR: missing %KERNEL_DEF% & exit /b 1)
if not exist "%BACKEND_SRC%" (echo ERROR: missing %BACKEND_SRC% & exit /b 1)
if not exist "%BACKEND_DEF%" (echo ERROR: missing %BACKEND_DEF% & exit /b 1)
if not exist "%BACKEND_SINGLE_DEF%" (echo ERROR: missing %BACKEND_SINGLE_DEF% & exit /b 1)

rem nvcc on Windows uses a supported MSVC host compiler for the .cu files.
where cl.exe >nul 2>&1
if errorlevel 1 (
  set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
  if exist "!VSWHERE!" (
    for /f "usebackq tokens=*" %%V in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSROOT=%%V"
    if defined VSROOT if exist "!VSROOT!\VC\Auxiliary\Build\vcvars64.bat" call "!VSROOT!\VC\Auxiliary\Build\vcvars64.bat" >nul
  )
)
where cl.exe >nul 2>&1 || (echo ERROR: cl.exe not found for nvcc compilation & exit /b 1)

rem Generate MinGW import libraries in addition to the MSVC .lib files emitted by nvcc.
set "DLLTOOL="
if not "%~2"=="" if exist "%~2\dlltool.exe" set "DLLTOOL=%~2\dlltool.exe"
if not defined DLLTOOL for /f "delims=" %%D in ('where dlltool.exe 2^>nul') do if not defined DLLTOOL set "DLLTOOL=%%D"
if not defined DLLTOOL (
  echo ERROR: MinGW dlltool.exe not found. Pass the MinGW bin directory as argument 2.
  exit /b 1
)

echo ============================================================
echo [CUDA 1/4] Split kernels - double precision
call :BUILD_KERNEL double "" adda_cuda_kernels
if errorlevel 1 exit /b 1

echo [CUDA 2/4] Split kernels - single precision
call :BUILD_KERNEL single "-DADDA_CUDA_SINGLE_BACKEND" adda_cuda_kernels_single
if errorlevel 1 exit /b 1

echo [CUDA 3/4] Monolithic backend - double precision
call :BUILD_BACKEND double "" adda_cuda_backend "%BACKEND_DEF%"
if errorlevel 1 exit /b 1

echo [CUDA 4/4] Monolithic backend - single precision
call :BUILD_BACKEND single "-DADDA_CUDA_SINGLE_BACKEND" adda_cuda_backend_single "%BACKEND_SINGLE_DEF%"
if errorlevel 1 exit /b 1

echo.
echo ============================================================
echo All CUDA libraries built successfully with %ARCHFLAG%
echo.
echo Split backend libraries:
echo   %KERNEL_OUT%\adda_cuda_kernels.dll
echo   %KERNEL_OUT%\libadda_cuda_kernels.a
echo   %KERNEL_OUT%\adda_cuda_kernels.lib
echo   %KERNEL_OUT%\adda_cuda_kernels_single.dll
echo   %KERNEL_OUT%\libadda_cuda_kernels_single.a
echo   %KERNEL_OUT%\adda_cuda_kernels_single.lib
echo.
echo Monolithic backend libraries:
echo   %BACKEND_OUT%\adda_cuda_backend.dll
echo   %BACKEND_OUT%\libadda_cuda_backend.a
echo   %BACKEND_OUT%\adda_cuda_backend.lib
echo   %BACKEND_OUT%\adda_cuda_backend_single.dll
echo   %BACKEND_OUT%\libadda_cuda_backend_single.a
echo   %BACKEND_OUT%\adda_cuda_backend_single.lib
echo ============================================================
exit /b 0

:BUILD_KERNEL
set "LABEL=%~1"
set "PRECDEF=%~2"
set "BASE=%~3"
set "DLL=%KERNEL_OUT%\%BASE%.dll"
set "MSVCLIB=%KERNEL_OUT%\%BASE%.lib"
set "GNULIB=%KERNEL_OUT%\lib%BASE%.a"

echo [CUDA kernels %LABEL%] nvcc %ARCHFLAG% -O3 %PRECDEF%
del /q "%DLL%" "%MSVCLIB%" "%GNULIB%" 2>nul
nvcc -std=c++14 --shared -O3 %ARCHFLAG% %PRECDEF% ^
  -I"%ROOT%\src" "%KERNEL_SRC%" ^
  -Xlinker "/DEF:%KERNEL_DEF%" ^
  -Xlinker "/IMPLIB:%MSVCLIB%" ^
  -o "%DLL%"
if errorlevel 1 (echo ERROR: nvcc failed for %LABEL% split kernels & exit /b 1)

"%DLLTOOL%" --def "%KERNEL_DEF%" --dllname "%BASE%.dll" --output-lib "%GNULIB%"
if errorlevel 1 (echo ERROR: dlltool failed for %LABEL% split kernels & exit /b 1)
if not exist "%DLL%" (echo ERROR: missing %DLL% & exit /b 1)
if not exist "%MSVCLIB%" (echo ERROR: missing %MSVCLIB% & exit /b 1)
if not exist "%GNULIB%" (echo ERROR: missing %GNULIB% & exit /b 1)
exit /b 0

:BUILD_BACKEND
set "LABEL=%~1"
set "PRECDEF=%~2"
set "BASE=%~3"
set "CURDEF=%~4"
set "DLL=%BACKEND_OUT%\%BASE%.dll"
set "MSVCLIB=%BACKEND_OUT%\%BASE%.lib"
set "GNULIB=%BACKEND_OUT%\lib%BASE%.a"

echo [CUDA monolithic %LABEL%] nvcc %ARCHFLAG% -O3 %PRECDEF%
del /q "%DLL%" "%MSVCLIB%" "%GNULIB%" 2>nul
nvcc -std=c++14 --shared -O3 %ARCHFLAG% %PRECDEF% ^
  -I"%ROOT%\src" "%BACKEND_SRC%" ^
  -lcufft -lcublas ^
  -Xlinker "/DEF:%CURDEF%" ^
  -Xlinker "/IMPLIB:%MSVCLIB%" ^
  -o "%DLL%"
if errorlevel 1 (echo ERROR: nvcc failed for %LABEL% monolithic backend & exit /b 1)

"%DLLTOOL%" --def "%CURDEF%" --dllname "%BASE%.dll" --output-lib "%GNULIB%"
if errorlevel 1 (echo ERROR: dlltool failed for %LABEL% monolithic backend & exit /b 1)
if not exist "%DLL%" (echo ERROR: missing %DLL% & exit /b 1)
if not exist "%MSVCLIB%" (echo ERROR: missing %MSVCLIB% & exit /b 1)
if not exist "%GNULIB%" (echo ERROR: missing %GNULIB% & exit /b 1)
exit /b 0
