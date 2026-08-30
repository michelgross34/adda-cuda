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
Sous Windows, nvcc doit utiliser un compilateur hote CUDA supporte (MSVC).
Ouvrez 'x64 Native Tools Command Prompt for Visual Studio', puis relancez PowerShell,
ou passez -HostCompiler avec le chemin du repertoire contenant cl.exe.
ADDA lui-meme n'est PAS compile avec MSVC : seul le fichier .cu passe par nvcc.
"@
    }
    $ccbin = Split-Path $cl.Source -Parent
}

$CuFile = Join-Path $Root "src\cudamatvec_backend.cu"
$DefFile = Join-Path $Root "src\cudamatvec_backend.def"
$IncludeDir = Join-Path $Root "src"
$BackendDll = Join-Path $Out "adda_cuda_backend.dll"
$BackendMsvcLib = Join-Path $Out "adda_cuda_backend.lib"
$BackendGnuLib = Join-Path $Out "libadda_cuda_backend.a"

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

Write-Host "[CUDA 1/2] Compilation du backend avec nvcc uniquement"
& $nvcc @nvccArgs
if ($LASTEXITCODE -ne 0) { throw "nvcc a echoue ($LASTEXITCODE)" }
if (-not (Test-Path $BackendDll)) { throw "DLL CUDA non produite: $BackendDll" }

Write-Host "[CUDA 2/2] Creation de l'import library GNU pour MinGW"
& $dlltool --def $DefFile --dllname "adda_cuda_backend.dll" --output-lib $BackendGnuLib
if ($LASTEXITCODE -ne 0) { throw "dlltool a echoue ($LASTEXITCODE)" }
if (-not (Test-Path $BackendGnuLib)) { throw "Import library MinGW non produite: $BackendGnuLib" }

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
Write-Host "Backend CUDA pret pour MinGW :"
Write-Host "  DLL       : $BackendDll"
Write-Host "  import lib: $BackendGnuLib"
Write-Host "CMake/CLion ne doit jamais compiler cudamatvec_backend.cu."
