# CUDA backends - build separe de CMake

`src/cudamatvec_backend.cu` ne doit **jamais** etre compile par CMake/CLion.

Le meme source CUDA est compile deux fois par `nvcc` :

1. backend float64 : `adda_cuda_backend.dll` ;
2. backend float32 : `adda_cuda_backend_single.dll` avec `-DADDA_CUDA_SINGLE_BACKEND`.

MinGW utilise respectivement :

- `libadda_cuda_backend.a`
- `libadda_cuda_backend_single.a`

Le backend float32 remplace les donnees/FFT/BLAS par `cuFloatComplex`, `CUFFT_C2C`, `cublasCdotu`, `cublasCdotc` et `cublasScnrm2` tout en conservant la meme ABI C.

## Methode recommandee

Depuis la racine :

```bat
scripts\build_cuda_backend.bat
```

Le script utilise `-arch=native` par defaut (CUDA 11.8 compatible) et produit les deux DLL et les deux import libraries. Ensuite CMake/CLion reste un projet C/MinGW pur.

## Cibles

Double : `adda_cuda`, `adda_cuda_slice`, `adda_low_mem_double`.

Float32 : `adda_cuda_single`, `adda_single_slice`, `adda_low_mem`.

Voir `README_SINGLE_PRECISION.md` pour les details et les dependances FFTW3F.
