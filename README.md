# ADDA-CUDA

CUDA acceleration of [ADDA](https://github.com/adda-team/adda), the light-scattering simulator based on the discrete dipole approximation (DDA).

This repository contains the CUDA port developed from the official ADDA source tree. It follows the same general approach as the CUDA work carried out for [DDSCAT](https://github.com/michelgross34/ddscat-cuda), while preserving the ADDA calculation pipeline and command-line interface.

ADDA calculates electromagnetic scattering and absorption by particles of arbitrary shape and composition. It can also handle particles near a plane substrate and calculate emission or decay-rate enhancement. See the [official ADDA project](https://github.com/adda-team/adda) and the [ADDA manual](doc/manual.pdf) for the physical model and complete list of options.

## Features

- CPU and NVIDIA CUDA implementations of the ADDA solver.
- Single precision (`float32`) and double precision (`float64`) executables.
- Full 3-D GPU FFT implementation.
- Slice-based GPU FFT implementation for reduced VRAM consumption.
- Low-memory CUDA organization for larger problems.
- CUDA-resident iterative solver vectors, avoiding CPU-GPU transfers during the QMR calculation.
- CUDA implementations of the available iterative-solver vector operations.
- Windows build based on Makefiles, MinGW, GCC, G++, and GFortran.
- Separate CUDA backend compilation with `nvcc`, using `scripts\\build_cuda_backend.bat`.

## Executables

The following executables and CUDA DLLs were present in the CLion Debug output directory `cmake-build-debug\\bin\\` on 28 August 2026.

| Executable | Processor | Precision | FFT / memory organization | Purpose |
|---|---|---|---|---|
| `adda.exe` | CPU | double / `float64` | Standard FFT | Reference ADDA implementation |
| `adda_single.exe` | CPU | single / `float32` | Standard FFT | CPU single-precision comparison |
| `adda_cuda.exe` | NVIDIA GPU | double / `float64` | Full 3-D GPU FFT | Accurate CUDA version; highest VRAM use |
| `adda_cuda_single.exe` | NVIDIA GPU | single / `float32` | Full 3-D GPU FFT | Lower-memory and faster CUDA version |
| `adda_cuda_slice.exe` | NVIDIA GPU | double / `float64` | Slice FFT | Reduced VRAM in double precision |
| `adda_single_slice.exe` | NVIDIA GPU | single / `float32` | Slice FFT | Lowest-memory slice variant |
| `adda_low_mem.exe` | NVIDIA GPU | single / `float32` | Low-memory layout | Minimum large resident GPU buffers |
| `adda_low_mem_double.exe` | NVIDIA GPU | double / `float64` | Low-memory layout | Low-memory layout with double precision |

The CUDA backend DLLs are:

| DLL | Precision | Role |
|---|---|---|
| `adda_cuda_backend.dll` | double / `float64` | CUDA backend for double-precision CUDA executables |
| `adda_cuda_backend_single.dll` | single / `float32` | CUDA backend for single-precision CUDA executables |

## CMake entry points

The root `CMakeLists.txt` contains the shared source lists and build rules. The
platform-specific files only select the appropriate options and call the root
configuration, so the source list is not duplicated:

```text
CMakeLists.txt        Shared CMake configuration
windows/CMakeLists.txt Windows / MinGW / CUDA entry point
linux/CMakeLists.txt   Linux / CPU entry point
```

Use the platform-specific entry point when configuring from a clean checkout.
Do not configure the Windows build from `linux/`, or the Linux build from
`windows/`.

Windows with MinGW (after generating the CUDA backends):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_cuda_backend.ps1 -CudaArch native
cmake -S windows -B build-windows -G Ninja
cmake --build build-windows --target adda_cuda_single -j 14
```

Linux uses the CPU targets and does not try to link the Windows CUDA DLL/import
libraries. FFTW libraries must be installed on the Linux system or supplied
through `FFTW3_ROOT`, `FFTW3_LIBRARY`, and `FFTW3F_LIBRARY`:

```bash
cmake -S linux -B build-linux -G Ninja
cmake --build build-linux -j
```

The root `CMakeLists.txt` remains the shared implementation used by both entry points.

## The `fftw/` directory

The repository root contains a Windows FFTW3 distribution in `fftw/`. CMake
detects this directory automatically when `fftw/fftw3.h` is present, so no
separate FFTW installation is needed for the standard Windows configuration.

The files most useful for building ADDA on Windows are:

| File | Purpose |
|---|---|
| `fftw3.h` | FFTW3 C header included by ADDA |
| `libfftw3.dll.a` | MinGW import library for double precision |
| `libfftw3-3.dll` | Runtime DLL for double-precision CPU/CUDA executables |
| `libfftw3f.dll.a` | MinGW import library for single precision |
| `libfftw3f-3.dll` | Runtime DLL for single-precision executables |
| `libfftw3.a`, `libfftw3f.a` | Static libraries, available for alternative link configurations |

The directory also contains FFTW Fortran interfaces, long-double/quadruple
precision variants, threaded/ OpenMP variants, and FFTW wisdom utilities. They
are not required by the default ADDA CMake targets. The CMake build copies the
appropriate `libfftw3-3.dll` or `libfftw3f-3.dll` next to each executable.

For a different FFTW installation, override the automatic detection, for
example:

```powershell
cmake -S windows -B build-windows -G Ninja `
  -DFFTW3_ROOT="C:/path/to/fftw" `
  -DFFTW3_LIBRARY="C:/path/to/fftw/libfftw3.dll.a" `
  -DFFTW3F_LIBRARY="C:/path/to/fftw/libfftw3f.dll.a"
```

On Linux, the files in this Windows-oriented `fftw/` directory should not be
used as native libraries. Install FFTW for Linux or provide Linux-compatible
paths through `FFTW3_ROOT`, `FFTW3_LIBRARY`, and `FFTW3F_LIBRARY`.

The names above describe the builds tested on Windows on 28 August 2026. The repository contains source code and build scripts; compiled binaries are not required to be committed to Git.

## Precision and convergence

Single precision reduces the storage required by complex vectors and FFT workspaces by approximately a factor of two. It is useful for problems that do not fit in GPU memory in double precision, but it also reduces the attainable numerical accuracy and can make iterative convergence more difficult.

Double precision is recommended for validation, difficult or poorly conditioned problems, and final production results. In single precision, `qmr2` is often preferable to `qmr` because it has better round-off properties. The final residual can be checked with:

```text
adda_cuda_single.exe ... -iter qmr2 -recalc_resid
```

For CUDA single-precision calculations, the `-reliable_resid` option periodically evaluates the true residual instead of relying only on the recursively estimated residual. It is intended to limit residual drift and improve convergence reliability. In the current implementation it applies to CUDA `float32` BiCGStab(2)/BCGS2 and GPBiCGStab(2):

```text
adda_cuda_single.exe ... -iter bcgs2 -reliable_resid
adda_cuda_single.exe ... -iter gpbicgstab2 -reliable_resid
```

The true residual is checked every 20 iterations. With `-reliable_resid`, the Krylov calculation is restarted only when the discrepancy indicates a significant residual error. With `-reliable_resid_force_restart`, a restart is forced at every 20-iteration check, even when the recursively estimated residual is still considered reliable; this option implies `-reliable_resid`.

Examples:

```text
adda_cuda_single.exe ... -iter bcgs2 -reliable_resid
adda_cuda_single.exe ... -iter bcgs2 -reliable_resid_force_restart
```

These options are ignored for unsupported solvers or for non-CUDA/non-single-precision builds. They can be combined with `-recalc_resid` when a final independent residual check is required.

## Iterative solvers

Compared with the original ADDA solver set, this version adds iterative solvers also available in [IFDDA](https://www.fresnel.fr/perso/chaumet/ifdda.html):

| Solver | Command-line option | Alias |
|---|---|---|
| Bi-CGStab(2) | `-iter bcgs2` | `-iter bicgstab2` |
| Bi-CG | `-iter bicg` | |
| Bi-CGStab | `-iter bicgstab` | |
| CGNR | `-iter cgnr` | |
| CSYM | `-iter csym` | |
| QMR | `-iter qmr` | |
| QMR2 | `-iter qmr2` | |
| BiCGStab(4) | `-iter bicgstab4` | |
| GPBiCGStab(2) | `-iter gpbicgstab2` | `-iter gpbicgstabl2` |
| GPBiCGStab(4) | `-iter gpbicgstab4` | `-iter gpbicgstabl4` |

The default solver remains `qmr`. For single-precision calculations, `qmr2` is a useful first choice when round-off limits convergence.

## Requirements on Windows

- 64-bit Windows.
- MinGW-w64 with GCC, G++, and GFortran.
- NVIDIA GPU and a compatible NVIDIA driver.
- NVIDIA CUDA Toolkit. CUDA compilation is performed separately with `nvcc`; the CUDA backend is then linked to the MinGW-built ADDA executable.
- FFTW development files for the CPU build, when required by the selected configuration.
- The Windows Makefiles in `src/` and `src/seq/`.

The ADDA C/C++/Fortran parts are compiled by the Windows Makefiles with MinGW, GCC, G++, and GFortran. The CUDA source is compiled separately with `nvcc`; it is not compiled as part of the MinGW ADDA build.

## Building

From a Windows command prompt, compile the CUDA backend first:

```bat
scripts\\build_cuda_backend.bat
```

The script invokes `nvcc` to create the CUDA DLL and MinGW `dlltool` to create its GNU import library. It does not compile ADDA itself. ADDA is then compiled separately with the Windows Makefile and linked against the CUDA backend import library. The script accepts an optional CUDA architecture and MinGW bin directory:

```bat
scripts\\build_cuda_backend.bat 86 "F:\\mingw64_11.2\\bin"
```

The Makefile is the authoritative Windows build entry point for the ADDA sources. CMake/CLion may be used to organize the project, but `nvcc` remains a separate build step.

## Precompiled Windows binaries

This repository includes a Windows Debug build in:

```text
windows/cmake-build-debug/bin/
```

The directory contains the ADDA executables, the FFTW runtime DLLs, and the
CUDA backend DLLs. This build was compiled with the CUDA 11.8 Toolkit and the
CUDA code was generated for compute capability 7.0 (`sm_70`), matching an
NVIDIA Titan V. It is therefore a ready-to-run reference build for that CUDA
environment and GPU generation, subject to a compatible NVIDIA driver.

The CUDA DLLs required by the executables are:

```text
cuda-backend/adda_cuda_backend.dll
cuda-backend/adda_cuda_backend_single.dll
```

At runtime, the appropriate CUDA DLL must be next to the executable in the
same directory. The Windows CMake post-build step copies these files to
`windows/cmake-build-debug/bin/`. If an executable is moved elsewhere, copy
the matching CUDA DLL from `cuda-backend/` beside it. The FFTW DLLs must also
remain beside the executable.

If the user has another CUDA Toolkit version or another GPU architecture, the
CUDA backend must be rebuilt locally. From the repository root, open an
environment where `nvcc.exe`, a supported MSVC host compiler (`cl.exe`), and
MinGW `dlltool.exe` are available, then run:

```bat
scripts\build_cuda_backend.bat 70 "F:\mingw64_11.2\bin"
```

Replace `70` with the target GPU compute capability, or omit the architecture
argument to use `-arch=native` when supported by the installed CUDA Toolkit.
The script builds both double-precision and float32 backends and writes the
new DLLs and GNU import libraries into `cuda-backend/`. Rebuild the ADDA
executables with the Windows CMake entry point afterwards:

```powershell
cmake -S windows -B build-windows -G Ninja
cmake --build build-windows -j 14
```

The generated `.a` import libraries are used for linking with MinGW; the DLLs
are needed at runtime and must be located beside the corresponding executable.

## Running ADDA

ADDA uses command-line options. For example, a 32-cubed grid and refractive index 1.6 can be requested with options of the following form:

```text
adda_cuda_single.exe -grid 32 -m 1.6 0 ...
```

The complete syntax, geometry options, material definitions, solver options, and output files are documented in [doc/manual.pdf](doc/manual.pdf). Use `-h` for the help available in the executable.

## Repository layout

```text
src/                       ADDA C sources and CUDA integration
src/cudamatvec_backend.cu CUDA backend and resident-vector operations
cuda-backend/              Separate CUDA backend build files
scripts/                   Windows MinGW/CUDA build scripts
tests/                     Regression and CUDA tests
third_party/               Required third-party headers
doc/                       ADDA documentation
```

## Relationship to ADDA and DDSCAT

ADDA-CUDA is based on the source code of the official [ADDA project](https://github.com/adda-team/adda). The CUDA implementation applies the same general GPU-acceleration effort developed for [DDSCAT-CUDA](https://github.com/michelgross34/ddscat-cuda), adapted to ADDA's C99 code, FFT pipeline, complex arithmetic, and iterative solvers.

For the original ADDA scientific references, capabilities, limitations, and licensing information, please consult the official ADDA repository and its documentation. This project is intended to remain compatible with the upstream ADDA command-line workflow wherever the CUDA implementation permits.

## License and citation

ADDA is distributed under the GNU General Public License. Please consult the license and attribution information in the upstream project before redistributing modified versions.

If results obtained with ADDA-CUDA are published, please cite the ADDA work and the relevant discrete-dipole-approximation references listed in the official documentation.

## Status

This repository is an experimental CUDA extension of ADDA. CPU and CUDA results should be compared on representative cases, especially when using single precision, slice FFT, or low-memory modes. Numerical agreement and convergence should be validated before using a configuration for production calculations.
