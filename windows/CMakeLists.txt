cmake_minimum_required(VERSION 3.20)

# Windows entry point.  The implementation and source lists remain in the
# repository-root CMakeLists.txt; this file only selects the Windows/MinGW
# configuration so the project can be configured from the windows directory.
project(ADDA_WINDOWS LANGUAGES C)

if(NOT WIN32)
    message(FATAL_ERROR "windows/CMakeLists.txt must be configured on Windows.")
endif()

set(ADDA_CUDA ON CACHE BOOL "Build double-precision CUDA targets" FORCE)
set(ADDA_CUDA_SLICE ON CACHE BOOL "Build double-precision CUDA slice target" FORCE)
set(ADDA_CUDA_LOW_MEM ON CACHE BOOL "Build double-precision low-memory CUDA target" FORCE)
set(ADDA_SINGLE ON CACHE BOOL "Build CPU single-precision target" FORCE)
set(ADDA_CUDA_SINGLE ON CACHE BOOL "Build float32 CUDA target" FORCE)
set(ADDA_CUDA_SINGLE_SLICE ON CACHE BOOL "Build float32 CUDA slice target" FORCE)
set(ADDA_CUDA_LOW_MEM_SINGLE ON CACHE BOOL "Build float32 low-memory CUDA target" FORCE)

set(_ADDA_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")
set(ADDA_CUDA_EXTERNAL_BACKEND
    "${_ADDA_ROOT}/cuda-backend/libadda_cuda_backend.a"
    CACHE FILEPATH "MinGW import library for the double CUDA backend" FORCE)
set(ADDA_CUDA_EXTERNAL_DLL
    "${_ADDA_ROOT}/cuda-backend/adda_cuda_backend.dll"
    CACHE FILEPATH "DLL for the double CUDA backend" FORCE)
set(ADDA_CUDA_SINGLE_EXTERNAL_BACKEND
    "${_ADDA_ROOT}/cuda-backend/libadda_cuda_backend_single.a"
    CACHE FILEPATH "MinGW import library for the float32 CUDA backend" FORCE)
set(ADDA_CUDA_SINGLE_EXTERNAL_DLL
    "${_ADDA_ROOT}/cuda-backend/adda_cuda_backend_single.dll"
    CACHE FILEPATH "DLL for the float32 CUDA backend" FORCE)

add_subdirectory("${_ADDA_ROOT}" "adda-core")
