param(
    [string]$BuildDir = "build-mingw-cuda",
    [ValidateSet("Release", "Debug")]
    [string]$Configuration = "Release",
    [string]$Generator = "Ninja",
    [string]$Gcc = "",
    [string]$FftwInclude = "",
    [string]$FftwLibrary = "",
    [string]$FftwSingleLibrary = "",
    [string]$CudaBackendDir = "cuda-backend"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Build = if ([System.IO.Path]::IsPathRooted($BuildDir)) { $BuildDir } else { Join-Path $Root $BuildDir }
$Backend = if ([System.IO.Path]::IsPathRooted($CudaBackendDir)) { $CudaBackendDir } else { Join-Path $Root $CudaBackendDir }

function Require-Command([string]$Name) {
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $cmd) { throw "Commande introuvable: $Name" }
    return $cmd.Source
}

$cmake = Require-Command "cmake.exe"
if (-not $Gcc) { $Gcc = Require-Command "gcc.exe" }

$BackendDll = Join-Path $Backend "adda_cuda_backend.dll"
$BackendLib = Join-Path $Backend "libadda_cuda_backend.a"
$BackendDllSingle = Join-Path $Backend "adda_cuda_backend_single.dll"
$BackendLibSingle = Join-Path $Backend "libadda_cuda_backend_single.a"
if (-not (Test-Path $BackendDll)) { throw "Backend CUDA absent: $BackendDll`nLancez d'abord scripts\build_cuda_backend.ps1" }
if (-not (Test-Path $BackendLib)) { throw "Import library MinGW absente: $BackendLib`nLancez d'abord scripts\build_cuda_backend.ps1" }
if (-not (Test-Path $BackendDllSingle)) { throw "Backend CUDA float32 absent: $BackendDllSingle`nLancez d'abord scripts\build_cuda_backend.ps1" }
if (-not (Test-Path $BackendLibSingle)) { throw "Import library MinGW float32 absente: $BackendLibSingle`nLancez d'abord scripts\build_cuda_backend.ps1" }

$cmakeArgs = @(
    "-S", $Root,
    "-B", $Build,
    "-G", $Generator,
    "-DADDA_CUDA=ON",
    "-DADDA_NO_FORTRAN=ON",
    "-DADDA_CUDA_EXTERNAL_BACKEND=$BackendLib",
    "-DADDA_CUDA_EXTERNAL_DLL=$BackendDll",
    "-DADDA_CUDA_SINGLE_EXTERNAL_BACKEND=$BackendLibSingle",
    "-DADDA_CUDA_SINGLE_EXTERNAL_DLL=$BackendDllSingle",
    "-DADDA_SINGLE=ON",
    "-DADDA_CUDA_SINGLE=ON",
    "-DADDA_CUDA_SINGLE_SLICE=ON",
    "-DADDA_CUDA_LOW_MEM_SINGLE=ON",
    "-DCMAKE_BUILD_TYPE=$Configuration",
    "-DCMAKE_C_COMPILER=$Gcc"
)
if ($FftwInclude) { $cmakeArgs += "-DFFTW3_INCLUDE_DIR=$FftwInclude" }
if ($FftwLibrary) { $cmakeArgs += "-DFFTW3_LIBRARY=$FftwLibrary" }
if ($FftwSingleLibrary) { $cmakeArgs += "-DFFTW3F_LIBRARY=$FftwSingleLibrary" }

Write-Host "[MinGW 1/2] Configuration CMake (C uniquement, aucun nvcc)"
& $cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "Configuration CMake a echoue ($LASTEXITCODE)" }

Write-Host "[MinGW 2/2] Compilation des cibles double et float32 avec MinGW"
& $cmake --build $Build --target adda adda_single adda_cuda adda_cuda_slice adda_low_mem_double adda_cuda_single adda_single_slice adda_low_mem -j
if ($LASTEXITCODE -ne 0) { throw "Build MinGW a echoue ($LASTEXITCODE)" }

Write-Host ""
Write-Host "Executables :"
Write-Host "  CPU double         : $(Join-Path $Build 'bin\adda.exe')"
Write-Host "  CPU float32        : $(Join-Path $Build 'bin\adda_single.exe')"
Write-Host "  CUDA double        : $(Join-Path $Build 'bin\adda_cuda.exe')"
Write-Host "  CUDA float32       : $(Join-Path $Build 'bin\adda_cuda_single.exe')"
Write-Host "  Slice float32      : $(Join-Path $Build 'bin\adda_single_slice.exe')"
Write-Host "  Low-memory float32 : $(Join-Path $Build 'bin\adda_low_mem.exe')"
Write-Host "  Low-memory double  : $(Join-Path $Build 'bin\adda_low_mem_double.exe')"
