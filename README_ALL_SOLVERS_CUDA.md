# Conversion CUDA de tous les solveurs itératifs ADDA

Cette version généralise le portage CUDA initial de `QMR_CS` aux sept solveurs présents dans `src/iterative.c` :

- `BCGS2` (`-iter bcgs2`)
- `BiCG_CS` (`-iter bicg`)
- `BiCGStab` (`-iter bicgstab`)
- `CGNR` (`-iter cgnr`)
- `CSYM` (`-iter csym`)
- `QMR_CS` (`-iter qmr`)
- `QMR_CS_2` (`-iter qmr2`)

## Principe conservé

Les algorithmes des solveurs restent en C dans `src/iterative.c`. Il n'existe pas de copie CUDA de chaque algorithme.

En compilation CPU, les macros de dispatch appellent les fonctions historiques de `linalg.c` et `MatVec()`.
En compilation `ADDA_CUDA`, les mêmes lignes C appellent :

- `MatVec_GPU()` pour les produits matrice-vecteur avec vecteurs déjà résidents ;
- cuBLAS, via les wrappers C, pour les produits scalaires et normes ;
- des kernels CUDA pour les opérations vectorielles élémentaires.

Cela maintient une seule source mathématique pour chaque solveur.

## MatVec et MatVec_GPU

`MatVec()` reste disponible et conserve le comportement :

1. copie H2D du vecteur d'entrée ;
2. appel du coeur MatVec CUDA ;
3. copie D2H du résultat.

`MatVec_GPU()` appelle directement le même coeur CUDA avec les pointeurs device associés aux vecteurs du solveur. Il ne fait aucune copie H2D/D2H des grands vecteurs.

Tous les solveurs CUDA utilisent `MatVec_GPU()` pendant leur boucle itérative.

## Registre de vecteurs GPU

Le backend peut enregistrer jusqu'à huit vecteurs logiques :

- `xvec`
- `rvec`
- `pvec`
- `vec1`
- `vec2`
- `vec3`
- `vec4`
- `Avecbuffer`

`BCGS2` est le cas maximal et utilise les huit.

Les buffers MatVec existants `d_arg` et `d_result` sont réutilisés respectivement pour `xvec` et `rvec`. Les autres vecteurs enregistrés sont placés dans une seule allocation contiguë `d_iter_extra`.

Les adresses des tableaux C servent uniquement d'identifiants. Les `SwapPointers()` de `iterative.c` restent donc inchangés : après un échange de pointeurs C, la recherche retrouve automatiquement le buffer CUDA correspondant à l'allocation désignée.

## Initialisation CUDA 11.8 et mémoire

L'ordre est :

1. `cudaGetDeviceCount()` / `cudaSetDevice()` ;
2. `cudaFree(0)` (`cudaFree(nullptr)` en C++) pour forcer la création du contexte sous CUDA 11.8 ;
3. création des événements ;
4. `cublasCreate()` et pointer mode HOST ;
5. création des plans cuFFT ;
6. allocations des matrices, grilles et buffers MatVec ;
7. au démarrage du solveur, allocation paresseuse du minimum de vecteurs supplémentaires requis.

Il n'existe pas de second contexte ou de seconde copie des matrices FFT pour les solveurs.

## Réductions cuBLAS

La convention ADDA est respectée :

- `nDotProd(a,b) = sum(a*conj(b))` -> `cublasZdotc(b,a)` ;
- `nDotProd_conj(a,b) = sum(a*b)` -> `cublasZdotu(a,b)` ;
- `nDotProdSelf_conj(a) = sum(a*a)` -> `cublasZdotu(a,a)` ;
- `nNorm2(a)` -> `cublasDznrm2(a)` puis carré ;
- `nDotProdSelf_conj_Norm2(a)` -> `cublasZdotu(a,a)` + `cublasDznrm2(a)`, puis carré de la norme.

Lorsqu'une opération vectorielle doit aussi retourner `|a|^2`, le kernel est lancé d'abord puis `cublasDznrm2()` est appelé sur le résultat.

## Kernels ajoutés/généralisés

Le backend couvre les primitives utilisées par tous les solveurs : copie, multiplication réelle/complexe, multiplication avec conjugaison, `LinComb`, `LinComb1`, variantes conjuguées, `Increm01`, `Increm10`, `Increm011`, `Increm110`, `Increm111`, `Increm11_d_c` et `Increm110_d_c_conj`.

## Checkpoints et retour CPU

Pendant la boucle itérative, la version device est la copie de référence.

Une synchronisation D2H de tous les vecteurs enregistrés est effectuée :

- avant `SaveIterChpoint()` ;
- une fois après la boucle itérative, avant `recalc_resid` et le post-traitement CPU.

Un checkpoint chargé est lu sur CPU puis les vecteurs sont uploadés une seule fois juste avant `PHASE_INIT`.

## Compilation Windows / CUDA 11.8

CMake reste C/MinGW uniquement et ne compile jamais le `.cu`.

Construire d'abord la DLL :

```bat
scripts\build_cuda_backend.bat
```

Sans argument, le script utilise `-arch=native`. Pour forcer une architecture :

```bat
scripts\build_cuda_backend.bat 86
```

Le backend est lié avec :

```text
-lcufft -lcublas
```

Puis reconstruire `adda_cuda` dans CLion/CMake afin que la nouvelle DLL soit recopiée à côté de l'exécutable.

## Validation réalisée dans l'environnement de génération

- syntaxe C99 de `iterative.c` en mode CPU : OK ;
- syntaxe C99 de `iterative.c` et `cudamatvec.c` avec `ADDA_CUDA` : OK ;
- compilation stricte `-Wall -Wextra -Werror` des fichiers C modifiés : OK ;
- build CMake complet des cibles `adda` et `adda_cuda` avec une DLL backend ABI factice : OK ;
- correspondance exacte des exports entre `cudamatvec_backend.h`, `.cu` et `.def` : OK ;
- pour les six nouveaux solveurs, après remplacement inverse des macros de dispatch, le texte de l'algorithme C est identique à la version précédente : OK.

L'environnement de génération ne contient pas `nvcc` ni de GPU NVIDIA. La compilation réelle de `cudamatvec_backend.cu` avec CUDA 11.8 et la validation numérique CPU/CUDA doivent être faites sur la machine cible.

## Ordre conseillé de validation numérique

Tester d'abord de petits cas identiques CPU/CUDA, solveur par solveur :

```text
bicg
bicgstab
cgnr
csym
qmr2
bcgs2
qmr
```

Pour chaque solveur, comparer au minimum : nombre d'itérations, trajectoire du résidu, résidu final recalculé et sorties physiques. Des écarts au dernier bit sont normaux car les réductions cuBLAS n'utilisent pas le même ordre de sommation que les boucles CPU.
