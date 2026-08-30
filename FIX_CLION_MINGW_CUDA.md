# Correctif CLion / MinGW / CUDA

## Principe retenu

- CMake/CLion compile uniquement les sources C d'ADDA avec MinGW/GCC.
- `nvcc` n'est jamais appelé par CMake.
- `src/cudamatvec_backend.cu` est compilé séparément en `cuda-backend/adda_cuda_backend.dll`.
- `dlltool` génère `cuda-backend/libadda_cuda_backend.a`, utilisée par l'édition de liens MinGW.
- `adda` reste la cible CPU.
- `adda_cuda` est la cible CUDA.

## Après remplacement du projet dans E:\msna6\adda-master

Votre ancien `cmake-build-debug` est incohérent (`CMakeFiles/rules.ninja` absent). Il faut le régénérer une fois :

```powershell
cd E:\msna6\adda-master
Remove-Item -Recurse -Force .\cmake-build-debug
```

Ou dans CLion : `Tools > CMake > Reset Cache and Reload Project`.

## Étape 1 — backend CUDA, séparément

Depuis un environnement où `nvcc.exe`, `cl.exe` et le `dlltool.exe` de MinGW sont accessibles :

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_cuda_backend.ps1 -CudaArch 86
```

Adapter `86` à votre GPU, ou supprimer `-CudaArch` pour laisser nvcc choisir sa valeur par défaut.

## Étape 2 — CLion / MinGW

Dans le profil CMake CLion :

```text
-DADDA_CUDA=ON -DADDA_NO_FORTRAN=ON
```

Si les fichiers ont été produits dans `cuda-backend`, aucun chemin CUDA supplémentaire n'est nécessaire.

Construire :

- `adda` pour la version CPU ;
- `adda_cuda` pour la version CUDA.

Important : la cible `adda` ne devient pas CUDA quand `ADDA_CUDA=ON`. L'option ajoute la cible `adda_cuda` sans modifier la cible CPU historique.
