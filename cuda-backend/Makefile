# Standalone CUDA-backend Makefile for Windows.
#
# IMPORTANT:
#   - This Makefile builds ONLY src/cudamatvec_backend.cu with nvcc.
#   - ADDA itself remains a MinGW/CMake build.
#   - nvcc on Windows uses MSVC (cl.exe) as its host compiler.
#   - The resulting DLL is exposed to MinGW through dlltool.
#
# Run from the project root, for example:
#   mingw32-make -C cuda-backend
# or explicitly: mingw32-make -C cuda-backend CUDA_ARCH=89
#
# cl.exe, nvcc.exe and dlltool.exe must be available in PATH.

NVCC      ?= nvcc
DLLTOOL   ?= dlltool
CUDA_ARCH ?= native

ifeq ($(CUDA_ARCH),native)
ARCH_FLAG := -arch=native
ARCH_LABEL := native
else ifeq ($(CUDA_ARCH),all)
ARCH_FLAG := -arch=all
ARCH_LABEL := all
else ifeq ($(CUDA_ARCH),all-major)
ARCH_FLAG := -arch=all-major
ARCH_LABEL := all-major
else
ARCH_FLAG := -arch=sm_$(CUDA_ARCH)
ARCH_LABEL := sm_$(CUDA_ARCH)
endif

ROOT      := ..
SRC       := $(ROOT)/src/cudamatvec_backend.cu
DEF       := $(ROOT)/src/cudamatvec_backend.def
INCLUDE   := $(ROOT)/src
DLL       := adda_cuda_backend.dll
MSVCLIB   := adda_cuda_backend.lib
GNULIB    := libadda_cuda_backend.a

.PHONY: all clean info

all: $(DLL) $(GNULIB)

$(DLL): $(SRC) $(DEF) $(ROOT)/src/cudamatvec_backend.h
	$(NVCC) -std=c++14 --shared -O3 $(ARCH_FLAG) -I"$(INCLUDE)" "$(SRC)" -lcufft -lcublas -Xlinker "/DEF:$(DEF)" -Xlinker "/IMPLIB:$(MSVCLIB)" -o "$(DLL)"

$(GNULIB): $(DLL) $(DEF)
	$(DLLTOOL) --def "$(DEF)" --dllname "$(DLL)" --output-lib "$(GNULIB)"

info:
	@echo CUDA DLL    : $(DLL)
	@echo MinGW import: $(GNULIB)
	@echo CUDA arch   : $(ARCH_LABEL)

clean:
	-del /Q "$(DLL)" "$(MSVCLIB)" "$(GNULIB)" 2>NUL
