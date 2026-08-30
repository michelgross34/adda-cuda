# Portage CUDA de `MatVec` — état du projet

## Résultat

Le portage CUDA du `MatVec` FFT séquentiel est implémenté.

La cible CPU historique reste **inchangée** et s'appelle `adda`. Lorsque CMake est configuré avec
`-DADDA_CUDA=ON`, une deuxième cible `adda_cuda` est ajoutée. Elle remplace uniquement
`src/matvec.c` par le wrapper `src/cudamatvec.c` et le backend CUDA
`src/cudamatvec_backend.cu`.

Le mode CUDA est volontairement séparé des modes `OPENCL`, `SPARSE` et `PARALLEL`; ces chemins
existants ne sont pas modifiés.

## Fichiers ajoutés

- `src/cudamatvec.c` : wrapper ADDA, timing, intégration aux variables globales.
- `src/cudamatvec.h` : points d'intégration C.
- `src/cudamatvec_backend.h` : ABI C entre le code C99 et le fichier `.cu`.
- `src/cudamatvec_backend.cu` : allocations CUDA persistantes, kernels et cuFFT.
- `src/cudamatvec_backend.def` : exports de la DLL pour le build mixte Windows/MinGW.
- `scripts/build_cuda_mingw.ps1` : build Windows MinGW + DLL CUDA `nvcc`.
- `tests/cuda_matvec_test.cu` : comparaison CUDA / référence DFT indépendante sur plusieurs grilles.

## Architecture numérique

Le CPU historique travaille tranche par tranche suivant `x` :

1. mise en grille et multiplication par `cc_sqrt`;
2. FFT selon `x`;
3. extraction d'une tranche `x`;
4. FFT `z`, transposition Y/Z, FFT `y`;
5. produit spectral par `Dmatrix` et éventuellement `Rmatrix`;
6. transformations inverses et restitution dans `Xmatrix`;
7. FFT inverse `x` puis reconstruction du vecteur résultat.

Le backend CUDA conserve à la place une **grille 3D complète** sur le GPU, en disposition
composante-major :

```
grid[c * (gridX*gridY*gridZ) + ((z*gridY + y)*gridX + x)]
```

Pour le terme direct `D`, un plan cuFFT 3D batched (3 composantes) remplace la succession des FFT
1D et les transpositions. Mathématiquement les transformations 1D commutent, et l'index spectral
reste exactement `(x,y,z)` comme dans `IndexDmatrix_mv`.

Pour le terme de surface `R`, le CPU calcule `Fx * Fy * Fz^-1(X)`. Le CUDA reproduit exactement ce
schéma avec un plan 2D XY en sens direct puis un plan Z stridé en sens inverse. Le produit `D*X +
R*XR` est ensuite soumis à la même FFT 3D inverse.

Les règles de symétrie de `reduced_FFT`, les changements de signe des composantes 12/13/23,
`IndexRmatrix_mv`, ainsi que le cas `her` / matrice transposée sont reproduits dans
`spectralMultiplyKernel`.

cuFFT, comme FFTW dans ADDA, est utilisé sans normalisation. Le facteur `-1/Ngrid` déjà incorporé
dans `Dmatrix` et `Rmatrix` par `InitDmatrix()` reste donc correct et aucun facteur supplémentaire
n'est appliqué.

## Gestion mémoire et transferts

Les objets suivants sont alloués / copiés **une seule fois** lors de `InitDmatrix()` :

- `Dmatrix`;
- `Rmatrix` si `surface=true`;
- `material`;
- `position`;
- grilles FFT et vecteurs de travail;
- plans cuFFT.

`cc_sqrt` est copié uniquement depuis `InitCC()`, car il peut changer entre deux calculs de
polarisation mais reste constant pendant les itérations du solveur.

À chaque appel `MatVec`, les seules copies PCIe obligatoires avec l'architecture ADDA actuelle sont :

- `argvec` : hôte -> GPU;
- `resultvec` : GPU -> hôte.

La copie retour est bloquante et constitue le point de synchronisation de l'appel. Les kernels et
cuFFT sont exécutés sur le flux CUDA par défaut dans l'ordre de soumission.

Les solveurs itératifs ont maintenant été portés sur des vecteurs CUDA résidents. Voir `README_ALL_SOLVERS_CUDA.md`.
Pour ces solveurs, `MatVec_GPU()` réutilise le même coeur MatVec sans les deux transferts de vecteurs par itération.

## Compilation CPU

Le comportement par défaut reste le CPU :

```bash
cmake -S . -B build -DADDA_NO_FORTRAN=ON
cmake --build build --config Release
```

Exécutable : `build/bin/adda` (`adda.exe` sous Windows).

## Compilation CUDA sous Windows : MinGW et nvcc strictement séparés

Le projet CMake est maintenant volontairement **C uniquement** :

```cmake
project(ADDA LANGUAGES C)
```

Même avec `-DADDA_CUDA=ON`, CMake **ne lance jamais `nvcc`**, n'active pas le langage CUDA et
ne compile aucun fichier `.cu`. Cette séparation est nécessaire pour une configuration CLion où
ADDA est compilé avec MinGW/GCC tandis que le CUDA Toolkit utilise son compilateur hôte Windows
supporté pour `nvcc`.

La construction se fait en deux étapes indépendantes.

### 1. Compiler le backend CUDA séparément avec nvcc

Ouvrir un environnement où `nvcc.exe`, `cl.exe` et `dlltool.exe` sont visibles, puis depuis la
racine du projet :

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_cuda_backend.ps1 -CudaArch 86
```

Cette commande ne compile **aucun fichier ADDA C**. Elle produit dans `cuda-backend\` :

- `adda_cuda_backend.dll` : DLL CUDA construite par `nvcc`;
- `adda_cuda_backend.lib` : import library MSVC utilisée uniquement par le test CUDA;
- `libadda_cuda_backend.a` : import library GNU créée par `dlltool`, destinée à MinGW.

`-CudaArch` est facultatif. Par exemple `86` correspond à `sm_86`.

### 2. Compiler ADDA avec MinGW / CLion

Dans CLion, utiliser une toolchain **MinGW**. CMake peut rester sur le générateur Ninja de CLion.
Les options CMake utiles sont :

```text
-DADDA_CUDA=ON
-DADDA_NO_FORTRAN=ON
-DADDA_CUDA_EXTERNAL_BACKEND=E:/msna6/adda-master/cuda-backend/libadda_cuda_backend.a
-DADDA_CUDA_EXTERNAL_DLL=E:/msna6/adda-master/cuda-backend/adda_cuda_backend.dll
```

Les deux dernières options sont facultatives si le backend est dans le dossier `cuda-backend`
de la racine, car ces chemins sont maintenant les valeurs par défaut.

CMake crée alors :

- `adda` : exécutable CPU historique, compilé uniquement avec MinGW;
- `adda_cuda` : exécutable ADDA utilisant le wrapper CUDA, lui aussi compilé avec MinGW et lié à
  `libadda_cuda_backend.a`.

Le fichier `src/cudamatvec_backend.cu` n'appartient à **aucune cible CMake**.

Pour construire en ligne de commande avec exactement le même découpage, après l'étape nvcc :

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_adda_mingw.ps1 `
  -BuildDir build-mingw-cuda `
  -Configuration Debug `
  -FftwInclude "C:\fftw\include" `
  -FftwLibrary "C:\fftw\lib\libfftw3-3.dll.a"
```

Ce second script ne cherche pas et n'appelle pas `nvcc`.

### Cas CLion : erreur `CMakeFiles/rules.ninja` absent

Une configuration CMake précédente qui a tenté d'activer CUDA peut laisser `cmake-build-debug`
incomplet. L'erreur :

```text
ninja: error: build.ninja:35: loading 'CMakeFiles/rules.ninja': GetLastError() = 2
```

signifie que le build Ninja généré est incohérent; ce n'est pas une erreur du code C.
Après remplacement par ce CMakeLists, supprimer **une seule fois** le dossier de build concerné,
puis laisser CLion le régénérer :

```powershell
Remove-Item -Recurse -Force .\cmake-build-debug
```

ou utiliser **Tools > CMake > Reset Cache and Reload Project** dans CLion.

Ensuite sélectionner la cible :

- `adda` pour le CPU;
- `adda_cuda` pour le MatVec CUDA.

Avec `ADDA_CUDA=ON`, construire seulement la cible `adda` ne nécessite ni la DLL CUDA ni `nvcc`.
La bibliothèque CUDA précompilée n'est requise qu'au moment de l'édition de liens de `adda_cuda`.

Pour forcer le GPU utilisé à l'exécution :

```text
ADDA_CUDA_DEVICE=0
```

## Tests numériques

Le test CUDA est lui aussi compilé séparément avec `nvcc` par `scripts/build_cuda_backend.ps1`.
Il compare le backend CUDA à une DFT 3D directe indépendante pour :

- grille `2x2x2`, `reduced_FFT`, sans surface;
- grille `4x4x4`, `reduced_FFT`, surface, `her=true`;
- grille `4x4x2`, FFT complète, surface, `her=true` (chemin transposé);
- grille `4x6x4`, FFT complète, sans surface.

Tolérances :

- absolue : `2e-9`;
- relative : `2e-10`.

Le test vérifie également que `argvec` n'est pas modifié et compare l'inner product / norme du
résultat. Sans GPU CUDA disponible, le script reconnaît le code retour `77` et signale simplement que le test GPU est ignoré.

## Validation effectuée dans l'environnement de conversion

- configuration et compilation complète de la cible CPU `adda` : **OK**;
- compilation syntaxique des fichiers C avec `ADDA_CUDA` : **OK**;
- analyse syntaxique CUDA/C++ du backend avec Clang CUDA : **OK**;
- comparaison mathématique indépendante entre l'algorithme historique par tranches et la nouvelle
  formulation grille-3D, y compris `surface`, `reduced_FFT` et `her` : écarts de l'ordre de
  `1e-13` en double précision sur les cas synthétiques testés;
- exécution réelle CUDA / cuFFT : **non effectuée dans cet environnement**, car `nvcc` et un GPU CUDA
  ne sont pas installés ici. Le test `cuda_matvec_test` est fourni pour cette validation sur la
  machine cible.

## Limites actuelles

- CUDA est séquentiel uniquement : pas de distribution MPI du MatVec CUDA dans ce patch.
- CUDA et OpenCL sont des backends distincts; aucun changement n'est apporté au chemin OpenCL.
- `SPARSE` continue d'utiliser son implémentation CPU spécifique.
- Les solveurs itératifs restent CPU-résidents, donc un aller-retour des vecteurs a lieu à chaque
  MatVec.
- La grille 3D CUDA complète consomme plus de mémoire GPU que le `Xmatrix` compact CPU. En contrepartie,
  elle supprime les tranches, transpositions et appels FFT répétés par `x`, ce qui est le compromis
  choisi pour ce premier backend CUDA robuste.

## Compilation Windows par fichier BAT

La compilation CUDA est maintenant accessible directement sans PowerShell :

```bat
scripts\build_cuda_backend.bat 86
```

Si `dlltool.exe` n'est pas dans le PATH :

```bat
scripts\build_cuda_backend.bat 86 "C:\chemin\vers\mingw\bin"
```

Le script produit `cuda-backend\adda_cuda_backend.dll` avec `nvcc`, puis `cuda-backend\libadda_cuda_backend.a` avec MinGW `dlltool`. CMake ne compile aucun fichier `.cu`.

Une variante Makefile autonome est egalement fournie :

```bat
mingw32-make -C cuda-backend CUDA_ARCH=86
```


## QMR_CS CUDA-résident

Le solveur `QMR_CS` reste en C dans `src/iterative.c`, mais ses opérations sur grands vecteurs utilisent désormais le backend CUDA résident. `MatVec_GPU()` évite les copies de vecteurs, les produits scalaires utilisent cuBLAS, et les combinaisons vectorielles utilisent des kernels CUDA. Voir `README_QMR_CUDA.md`.
