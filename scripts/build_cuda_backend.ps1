param(
    [string]$OutputDir = "cuda-backend",
    [ValidateSet("Release", "Debug")]
    [string]$Configuration = "Release",
    [string]$CudaArch = "native",
    [string]$HostCompiler = "",
    [switch]$SkipTest
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Out = if ([System.IO.Path]::IsPathRooted($OutputDir)) { $OutputDir } else { Join-Path $Root $OutputDir }
New-Item -ItemType Directory -Force -Path $Out | Out-Null

function Require-Command([string]$Name) {
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $cmd) { throw "Commande introuvable: $Name" }
    return $cmd.Source
}

$nvcc = Require-Command "nvcc.exe"
$dlltool = Require-Command "dlltool.exe"

if ($HostCompiler) {
    if (-not (Test-Path $HostCompiler)) { throw "Compilateur hote CUDA introuvable: $HostCompiler" }
    $ccbin = $HostCompiler
} else {
    $cl = Get-Command "cl.exe" -ErrorAction SilentlyContinue
if (-not $cl) {
    throw @"
cl.exe est introuvable.
CUDA peut etre installe sans Visual Studio, mais la compilation d'une DLL CUDA
avec nvcc sous Windows necessite un compilateur hote compatible.
Installez Visual Studio Build Tools avec la charge « Desktop development with C++ »,
puis ouvrez « x64 Native Tools Command Prompt for VS » avant de relancer ce script.
ou passez -HostCompiler avec le chemin du repertoire contenant cl.exe.
ADDA lui-meme n'est PAS compile avec MSVC : seul le fichier .cu passe par nvcc.
"@
    }
    $ccbin = Split-Path $cl.Source -Parent
}

$CuFile = Join-Path $Root "src\cudamatvec_backend.cu"
$DefFile = Join-Path $Root "src\cudamatvec_backend.def"
$DefFileSingle = Join-Path $Root "src\cudamatvec_backend_single.def"
$IncludeDir = Join-Path $Root "src"
$BackendDll = Join-Path $Out "adda_cuda_backend.dll"
$BackendMsvcLib = Join-Path $Out "adda_cuda_backend.lib"
$BackendGnuLib = Join-Path $Out "libadda_cuda_backend.a"
$BackendDllSingle = Join-Path $Out "adda_cuda_backend_single.dll"
$BackendMsvcLibSingle = Join-Path $Out "adda_cuda_backend_single.lib"
$BackendGnuLibSingle = Join-Path $Out "libadda_cuda_backend_single.a"

$nvccArgs = @(
    "-std=c++14",
    "--shared",
    "-ccbin", $ccbin,
    "-I$IncludeDir",
    $CuFile,
    "-lcufft",
    "-lcublas",
    "-Xlinker", "/DEF:$DefFile",
    "-Xlinker", "/IMPLIB:$BackendMsvcLib",
    "-o", $BackendDll
)

if ($Configuration -eq "Debug") {
    $nvccArgs = @("-G", "-O0", "-Xcompiler", "/MDd,/Od") + $nvccArgs
} else {
    $nvccArgs = @("-O3", "-Xcompiler", "/MD,/O2") + $nvccArgs
}
if ($CudaArch) {
    if ($CudaArch -ieq "native") {
        $nvccArgs = @("-arch=native") + $nvccArgs
    } else {
        $arch = $CudaArch -replace '^sm_', ''
        $nvccArgs = @("-arch=sm_$arch") + $nvccArgs
    }
}

Write-Host "[CUDA 1/4] Compilation du backend double avec nvcc uniquement"
& $nvcc @nvccArgs
if ($LASTEXITCODE -ne 0) { throw "nvcc a echoue ($LASTEXITCODE)" }
if (-not (Test-Path $BackendDll)) { throw "DLL CUDA non produite: $BackendDll" }

Write-Host "[CUDA 2/4] Creation de l'import library GNU double pour MinGW"
& $dlltool --def $DefFile --dllname "adda_cuda_backend.dll" --output-lib $BackendGnuLib
if ($LASTEXITCODE -ne 0) { throw "dlltool a echoue ($LASTEXITCODE)" }
if (-not (Test-Path $BackendGnuLib)) { throw "Import library MinGW non produite: $BackendGnuLib" }

Write-Host "[CUDA 3/4] Compilation du backend float32 avec nvcc"
$singleArgs = $nvccArgs.Clone()
# Rebuild arguments explicitly to avoid accidentally keeping double output/DEF paths.
$singleArgs = @(
    "-std=c++14", "--shared", "-ccbin", $ccbin, "-DADDA_CUDA_SINGLE_BACKEND",
    "-I$IncludeDir", $CuFile, "-lcufft", "-lcublas",
    "-Xlinker", "/DEF:$DefFileSingle", "-Xlinker", "/IMPLIB:$BackendMsvcLibSingle",
    "-o", $BackendDllSingle
)
if ($Configuration -eq "Debug") { $singleArgs = @("-G", "-O0", "-Xcompiler", "/MDd,/Od") + $singleArgs }
else { $singleArgs = @("-O3", "-Xcompiler", "/MD,/O2") + $singleArgs }
if ($CudaArch) {
    if ($CudaArch -ieq "native") { $singleArgs = @("-arch=native") + $singleArgs }
    else { $arch = $CudaArch -replace '^sm_', ''; $singleArgs = @("-arch=sm_$arch") + $singleArgs }
}
& $nvcc @singleArgs
if ($LASTEXITCODE -ne 0) { throw "nvcc float32 a echoue ($LASTEXITCODE)" }
if (-not (Test-Path $BackendDllSingle)) { throw "DLL CUDA float32 non produite: $BackendDllSingle" }

Write-Host "[CUDA 4/4] Creation de l'import library GNU float32"
& $dlltool --def $DefFileSingle --dllname "adda_cuda_backend_single.dll" --output-lib $BackendGnuLibSingle
if ($LASTEXITCODE -ne 0) { throw "dlltool float32 a echoue ($LASTEXITCODE)" }
if (-not (Test-Path $BackendGnuLibSingle)) { throw "Import library MinGW float32 non produite: $BackendGnuLibSingle" }

if (-not $SkipTest) {
    $TestCu = Join-Path $Root "tests\cuda_matvec_test.cu"
    $TestExe = Join-Path $Out "cuda_matvec_test.exe"
    if (Test-Path $BackendMsvcLib) {
        Write-Host "[CUDA test] Compilation du test avec nvcc"
        $testArgs = @("-std=c++14", "-ccbin", $ccbin, "-I$IncludeDir", $TestCu, $BackendMsvcLib, "-o", $TestExe)
        if ($Configuration -eq "Debug") { $testArgs = @("-G", "-O0") + $testArgs } else { $testArgs = @("-O2") + $testArgs }
        if ($CudaArch) {
            if ($CudaArch -ieq "native") {
                $testArgs = @("-arch=native") + $testArgs
            } else {
                $arch = $CudaArch -replace '^sm_', ''
                $testArgs = @("-arch=sm_$arch") + $testArgs
            }
        }
        & $nvcc @testArgs
        if ($LASTEXITCODE -ne 0) { throw "Compilation du test CUDA a echoue ($LASTEXITCODE)" }
        & $TestExe
        if ($LASTEXITCODE -eq 77) {
            Write-Warning "Aucun GPU CUDA disponible : test ignore (code 77)."
        } elseif ($LASTEXITCODE -ne 0) {
            throw "Le test CUDA a echoue ($LASTEXITCODE)"
        }
    } else {
        Write-Warning "adda_cuda_backend.lib absent : test CUDA non compile."
    }
}

Write-Host ""
Write-Host "Backends CUDA prets pour MinGW :"
Write-Host "  double DLL       : $BackendDll"
Write-Host "  double import lib: $BackendGnuLib"
Write-Host "  float32 DLL      : $BackendDllSingle"
Write-Host "  float32 import lib: $BackendGnuLibSingle"
Write-Host "CMake/CLion ne doit jamais compiler cudamatvec_backend.cu."
