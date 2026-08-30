cmake_minimum_required(VERSION 3.20)

# Linux entry point.  CUDA backends in this repository are Windows DLLs, so
# Linux builds use the portable CPU targets by default.
project(ADDA_LINUX LANGUAGES C)

if(WIN32)
    message(FATAL_ERROR "linux/CMakeLists.txt must be configured on Linux.")
endif()

set(ADDA_CUDA OFF CACHE BOOL "Build CUDA targets" FORCE)
set(ADDA_CUDA_SLICE OFF CACHE BOOL "Build CUDA slice target" FORCE)
set(ADDA_CUDA_LOW_MEM OFF CACHE BOOL "Build CUDA low-memory target" FORCE)
set(ADDA_CUDA_SINGLE OFF CACHE BOOL "Build float32 CUDA target" FORCE)
set(ADDA_CUDA_SINGLE_SLICE OFF CACHE BOOL "Build float32 CUDA slice target" FORCE)
set(ADDA_CUDA_LOW_MEM_SINGLE OFF CACHE BOOL "Build float32 CUDA low-memory target" FORCE)
set(ADDA_SINGLE ON CACHE BOOL "Build CPU single-precision target" FORCE)

set(_ADDA_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")
add_subdirectory("${_ADDA_ROOT}" "adda-core")
