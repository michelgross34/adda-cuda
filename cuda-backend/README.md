# CUDA backend - build separe de CMake

Le fichier `src/cudamatvec_backend.cu` ne doit **jamais** etre compile par CMake/CLion.

Sous Windows la chaine est volontairement separee :

1. `nvcc` compile le backend CUDA dans `adda_cuda_backend.dll` (avec `cl.exe` comme compilateur hote supporte par CUDA) ;
2. MinGW `dlltool` cree `libadda_cuda_backend.a`, une petite bibliotheque d'import GNU ;
3. CMake/CLion compile ensuite ADDA avec MinGW et lie `adda_cuda` contre cette bibliotheque d'import.

## Methode recommandee : BAT

Depuis la racine du projet :

```bat
scripts\build_cuda_backend.bat
```

Sans argument, le script utilise `nvcc -arch=native`: nvcc detecte le GPU installe et genere le code pour cette architecture. C'est le mode recommande si la compilation et l'execution ont lieu sur la meme machine. Pour forcer une architecture, utiliser par exemple `scripts\build_cuda_backend.bat 89`.

Si `dlltool.exe` n'est pas dans le `PATH`, passer le repertoire `bin` MinGW en deuxieme argument :

```bat
scripts\build_cuda_backend.bat "C:\msys64\mingw64\bin"
```

Avec le MinGW fourni par CLion, indiquer de la meme maniere son repertoire `bin`.

Le script recherche automatiquement Visual Studio Build Tools avec `vswhere` lorsque `cl.exe` n'est pas deja actif.

## Methode Makefile

Ouvrir un shell dans lequel `nvcc.exe`, `cl.exe` et `dlltool.exe` sont disponibles, puis :

```bat
mingw32-make -C cuda-backend
```

## Commandes manuelles equivalentes

Depuis la racine du projet :

```bat
if not exist cuda-backend mkdir cuda-backend
nvcc -std=c++14 --shared -O3 -arch=native -Isrc src\cudamatvec_backend.cu -lcufft -lcublas -Xlinker /DEF:src\cudamatvec_backend.def -Xlinker /IMPLIB:cuda-backend\adda_cuda_backend.lib -o cuda-backend\adda_cuda_backend.dll
dlltool --def src\cudamatvec_backend.def --dllname adda_cuda_backend.dll --output-lib cuda-backend\libadda_cuda_backend.a
```

Fichiers obtenus :

- `cuda-backend/adda_cuda_backend.dll` : vrai code CUDA produit par `nvcc` ;
- `cuda-backend/adda_cuda_backend.lib` : import library MSVC, utile pour les tests compiles par `nvcc` ;
- `cuda-backend/libadda_cuda_backend.a` : import library GNU utilisee par MinGW/CLion.

Ensuite seulement, configurer CLion/CMake avec :

```text
-DADDA_CUDA=ON
```

et compiler la cible `adda_cuda`. La DLL est copiee a cote de l'executable par CMake.


## Erreur `no kernel image is available`

Cette erreur signifie que la DLL chargee ne contient pas de code GPU compatible avec le GPU courant. Recompiler avec `scripts\build_cuda_backend.bat` (sans numero, donc `-arch=native`), puis **recompiler la cible `adda_cuda` dans CLion** afin que CMake recopie la nouvelle DLL dans `cmake-build-debug\bin`. On peut aussi copier manuellement `cuda-backend\adda_cuda_backend.dll` a cote de `adda_cuda.exe`.
