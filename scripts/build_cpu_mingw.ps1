param(
    [string]$BuildDir = "build-mingw-cpu",
    [ValidateSet("Release", "Debug")]
    [string]$Configuration = "Release",
    [string]$Generator = "Ninja",
    [string]$Gcc = "",
    [string]$FftwInclude = "",
    [string]$FftwLibrary = "",
    [string]$FftwSingleLibrary = ""
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Build = if ([System.IO.Path]::IsPathRooted($BuildDir)) { $BuildDir } else { Join-Path $Root $BuildDir }

function Require-Command([string]$Name) {
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $cmd) { throw "Commande introuvable: $Name" }
    return $cmd.Source
}

$cmake = Require-Command "cmake.exe"
if (-not $Gcc) { $Gcc = Require-Command "gcc.exe" }

$cmakeArgs = @(
    "-S", $Root,
    "-B", $Build,
    "-G", $Generator,
    "-DADDA_CUDA=OFF",
    "-DADDA_CUDA_SLICE=OFF",
    "-DADDA_CUDA_LOW_MEM=OFF",
    "-DADDA_CUDA_SINGLE=OFF",
    "-DADDA_CUDA_SINGLE_SLICE=OFF",
    "-DADDA_CUDA_LOW_MEM_SINGLE=OFF",
    "-DADDA_SINGLE=ON",
    "-DADDA_NO_FORTRAN=ON",
    "-DADDA_NO_GITHASH=ON",
    "-DCMAKE_BUILD_TYPE=$Configuration",
    "-DCMAKE_C_COMPILER=$Gcc"
)
if ($FftwInclude) { $cmakeArgs += "-DFFTW3_INCLUDE_DIR=$FftwInclude" }
if ($FftwLibrary) { $cmakeArgs += "-DFFTW3_LIBRARY=$FftwLibrary" }
if ($FftwSingleLibrary) { $cmakeArgs += "-DFFTW3F_LIBRARY=$FftwSingleLibrary" }

Write-Host "[CPU 1/2] Configuration CMake MinGW"
& $cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "Configuration CMake a echoue ($LASTEXITCODE)" }

Write-Host "[CPU 2/2] Compilation adda.exe + adda_single.exe"
& $cmake --build $Build --target adda adda_single -j
if ($LASTEXITCODE -ne 0) { throw "Build CPU a echoue ($LASTEXITCODE)" }

Write-Host ""
Write-Host "Executables :"
Write-Host "  CPU double  : $(Join-Path $Build 'bin\adda.exe')"
Write-Host "  CPU float32 : $(Join-Path $Build 'bin\adda_single.exe')"
Write-Host ""
Write-Host "Solveurs :"
Write-Host "  -iter bicgstab2     (BCGS2 / BiCGStab(2))"
Write-Host "  -iter gpbicgstab2   (GPBiCGStab(2))"
Write-Host "  -iter bicgstab4     (BiCGStab(L=4), CPU only)"
Write-Host "  -iter gpbicgstab4   (GPBiCGStab(L=4), CPU only)"
