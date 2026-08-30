cmake_minimum_required(VERSION 3.20)

# Linux entry point. CUDA is built separately into native shared libraries by
# scripts/build_cuda_backend.sh, then all CPU and CUDA variants are built here.
project(ADDA_LINUX LANGUAGES C)

if(WIN32)
    message(FATAL_ERROR "linux/CMakeLists.txt must be configured on Linux.")
endif()

set(ADDA_CUDA ON CACHE BOOL "Build CUDA targets" FORCE)
set(ADDA_CUDA_SLICE ON CACHE BOOL "Build CUDA slice target" FORCE)
set(ADDA_CUDA_LOW_MEM ON CACHE BOOL "Build CUDA low-memory target" FORCE)
set(ADDA_CUDA_SINGLE ON CACHE BOOL "Build float32 CUDA target" FORCE)
set(ADDA_CUDA_SINGLE_SLICE ON CACHE BOOL "Build float32 CUDA slice target" FORCE)
set(ADDA_CUDA_LOW_MEM_SINGLE ON CACHE BOOL "Build float32 CUDA low-memory target" FORCE)
set(ADDA_SINGLE ON CACHE BOOL "Build CPU single-precision target" FORCE)
set(ADDA_NO_GITHASH ON CACHE BOOL "ZIP builds do not embed a Git revision" FORCE)

set(_ADDA_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")
set(FFTW3_ROOT "" CACHE PATH "Use native Linux FFTW3" FORCE)
set(FFTW3_INCLUDE_DIR "" CACHE PATH "Find the native Linux FFTW3 header" FORCE)
set(FFTW3_LIBRARY "" CACHE FILEPATH "Find native Linux FFTW3 double" FORCE)
set(FFTW3F_LIBRARY "" CACHE FILEPATH "Find native Linux FFTW3 float" FORCE)
set(ADDA_CUDA_EXTERNAL_BACKEND
    "${_ADDA_ROOT}/cuda-backend/libadda_cuda_backend.so"
    CACHE FILEPATH "Native Linux double CUDA backend" FORCE)
set(ADDA_CUDA_EXTERNAL_DLL
    "${_ADDA_ROOT}/cuda-backend/libadda_cuda_backend.so"
    CACHE FILEPATH "Native Linux double CUDA runtime" FORCE)
set(ADDA_CUDA_SINGLE_EXTERNAL_BACKEND
    "${_ADDA_ROOT}/cuda-backend/libadda_cuda_backend_single.so"
    CACHE FILEPATH "Native Linux float32 CUDA backend" FORCE)
set(ADDA_CUDA_SINGLE_EXTERNAL_DLL
    "${_ADDA_ROOT}/cuda-backend/libadda_cuda_backend_single.so"
    CACHE FILEPATH "Native Linux float32 CUDA runtime" FORCE)
add_subdirectory("${_ADDA_ROOT}" "adda-core")
