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
