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
- MinGW/GCC C99 build support on Windows.

## Executables

The following executables are produced by the Windows build. They are normally found in `cmake-build-debug/bin` or in the selected CMake build directory.

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

The names above describe the builds tested on Windows on 28 August 2026. The repository contains source code and build scripts; compiled binaries are not required to be committed to Git.

## Precision and convergence

Single precision reduces the storage required by complex vectors and FFT workspaces by approximately a factor of two. It is useful for problems that do not fit in GPU memory in double precision, but it also reduces the attainable numerical accuracy and can make iterative convergence more difficult.

Double precision is recommended for validation, difficult or poorly conditioned problems, and final production results. In single precision, `qmr2` is often preferable to `qmr` because it has better round-off properties. The final residual can be checked with:

```text
adda_cuda_single.exe ... -iter qmr2 -recalc_resid
```

Available iterative solvers are `bcgs2`, `bicg`, `bicgstab`, `cgnr`, `csym`, `qmr`, and `qmr2`. The default solver is `qmr`.

## Requirements on Windows

- 64-bit Windows.
- MinGW-w64 with GCC and, for the CPU/Fortran parts where required, GFortran.
- NVIDIA GPU and a compatible NVIDIA driver.
- NVIDIA CUDA Toolkit. CUDA compilation is performed separately with `nvcc`; the CUDA backend is then linked to the MinGW-built ADDA executable.
- FFTW development files for the CPU build, when required by the selected configuration.
- CMake and Ninja, or the provided PowerShell/batch build scripts.

The scripts in `scripts/` document the expected MinGW and CUDA build workflow. The CUDA backend is kept separate from the C/GFortran compilation because `nvcc` and the MinGW C toolchain must not compile the same source unit.

## Building

From a MinGW-enabled PowerShell terminal, configure and build the standard CMake project, for example:

```powershell
cmake -S . -B build-cuda -G Ninja `
  -DCMAKE_C_COMPILER="F:/mingw64_11.2/bin/gcc.exe" `
  -DADDA_CUDA=ON `
  -DADDA_NO_FORTRAN=ON `
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-cuda
```

The CUDA backend can also be built separately using the scripts in `scripts/` and the makefile in `cuda-backend/`. Consult the accompanying build documentation before selecting a CUDA architecture or CUDA Toolkit version.

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
