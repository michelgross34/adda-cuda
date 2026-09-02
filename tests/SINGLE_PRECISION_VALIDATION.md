# Validation performed in this environment

## C compilation

The following source configurations passed strict C syntax checks (`-Wall -Wextra -Werror`, optional unrelated warnings disabled):

- `ADDA_SINGLE` CPU
- `ADDA_SINGLE + ADDA_CUDA`
- `ADDA_SINGLE + ADDA_CUDA + ADDA_CUDA_SLICE`
- `ADDA_SINGLE + ADDA_CUDA + ADDA_CUDA_SLICE + ADDA_CUDA_LOW_MEM`

A complete CMake build with ABI/FFTW test libraries also linked all targets:

- `adda`
- `adda_single`
- `adda_cuda`
- `adda_cuda_slice`
- `adda_low_mem_double`
- `adda_cuda_single`
- `adda_single_slice`
- `adda_low_mem`

## CPU numerical smoke test

Command: `-grid 8 8 8 -maxiter 20`.

Both double and single converged at `RE_010` (10 iterations). Relative differences in the residual history ranged from about `2.5e-8` initially to about `3.2e-4` at the final residual, which is consistent with a float32 data path.

Double: `Qext = 1.078846014`.
Single: `Qext = 1.078846722`.

For this nominally nonabsorbing test, double gave `Cabs ~ 2.1e-17`, while single gave `Cabs ~ -7.6e-9`; the latter is a small signed float32 roundoff around physical zero.

## CUDA source preprocessing

With `ADDA_CUDA_SINGLE_BACKEND`, the CUDA source preprocesses to the intended single calls/types:

- `cuFloatComplex`
- `CUFFT_C2C`
- `cufftExecC2C`
- `cublasCdotu`
- `cublasCdotc`
- `cublasScnrm2`

No real `nvcc`/GPU execution was available in this environment. The final CUDA 11.8 compile/run must therefore be performed on the target NVIDIA machine.
