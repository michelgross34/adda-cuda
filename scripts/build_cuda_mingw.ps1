param(
    [string]$BuildDir = "build-mingw-cuda",
    [ValidateSet("Release", "Debug")]
    [string]$Configuration = "Release",
    [string]$CudaArch = "",
    [string]$FftwInclude = "",
    [string]$FftwLibrary = ""
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

Write-Warning "Ce script est seulement un raccourci. Les deux compilations restent separees : nvcc produit d'abord la DLL CUDA, puis CMake/MinGW compile ADDA. Pour CLion, utilisez de preference build_cuda_backend.ps1 puis laissez CLion compiler adda_cuda."

& (Join-Path $PSScriptRoot "build_cuda_backend.ps1") -OutputDir "cuda-backend" -Configuration $Configuration -CudaArch $CudaArch
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$mingwArgs = @{
    BuildDir = $BuildDir
    Configuration = $Configuration
    CudaBackendDir = "cuda-backend"
}
if ($FftwInclude) { $mingwArgs["FftwInclude"] = $FftwInclude }
if ($FftwLibrary) { $mingwArgs["FftwLibrary"] = $FftwLibrary }
& (Join-Path $PSScriptRoot "build_adda_mingw.ps1") @mingwArgs
exit $LASTEXITCODE
