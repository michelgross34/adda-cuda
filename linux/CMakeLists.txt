cmake_minimum_required(VERSION 3.20)

# Linux entry point. CUDA is built separately into native shared libraries by
# scripts/build_cuda_backends.sh, then all CPU and CUDA variants are built here.
project(ADDA_LINUX LANGUAGES C)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
   message(FATAL_ERROR "linux/CMakeLists.txt must be configured for Linux.")
endif()

option(ADDA_CUDA_SPLIT_BACKEND
       "Use kernel.cu with the CMake-compiled C++ wrapper backend"
       ON)
option(ADDA_LINUX_CUDA "Build all Linux CUDA targets" ON)
# This entry point intentionally keeps every CUDA variant in sync. Configure
# the root CMakeLists.txt directly when individual CUDA targets are required.
set(ADDA_CUDA "${ADDA_LINUX_CUDA}" CACHE BOOL "Build CUDA targets" FORCE)
set(ADDA_CUDA_SLICE "${ADDA_LINUX_CUDA}" CACHE BOOL "Build CUDA slice target" FORCE)
set(ADDA_CUDA_LOW_MEM "${ADDA_LINUX_CUDA}" CACHE BOOL "Build CUDA low-memory target" FORCE)
set(ADDA_CUDA_SINGLE "${ADDA_LINUX_CUDA}" CACHE BOOL "Build float32 CUDA target" FORCE)
set(ADDA_CUDA_SINGLE_SLICE "${ADDA_LINUX_CUDA}" CACHE BOOL "Build float32 CUDA slice target" FORCE)
set(ADDA_CUDA_LOW_MEM_SINGLE "${ADDA_LINUX_CUDA}" CACHE BOOL "Build float32 CUDA low-memory target" FORCE)
set(ADDA_SINGLE ON CACHE BOOL "Build CPU single-precision target")
set(ADDA_NO_GITHASH ON CACHE BOOL "Do not embed a Git revision")

set(_ADDA_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")
if(CMAKE_BUILD_TYPE MATCHES "^[Dd]ebug$")
    set(_ADDA_CUDA_CONFIG_DIR "debug")
else()
    set(_ADDA_CUDA_CONFIG_DIR "release")
endif()
set(ADDA_CUDA_EXTERNAL_BACKEND
    "${_ADDA_ROOT}/cuda-backend/${_ADDA_CUDA_CONFIG_DIR}/libadda_cuda_backend.so"
    CACHE FILEPATH "Native Linux double CUDA backend")
set(ADDA_CUDA_EXTERNAL_DLL
    "${_ADDA_ROOT}/cuda-backend/${_ADDA_CUDA_CONFIG_DIR}/libadda_cuda_backend.so"
    CACHE FILEPATH "Native Linux double CUDA runtime")
set(ADDA_CUDA_SINGLE_EXTERNAL_BACKEND
    "${_ADDA_ROOT}/cuda-backend/${_ADDA_CUDA_CONFIG_DIR}/libadda_cuda_backend_single.so"
    CACHE FILEPATH "Native Linux float32 CUDA backend")
set(ADDA_CUDA_SINGLE_EXTERNAL_DLL
    "${_ADDA_ROOT}/cuda-backend/${_ADDA_CUDA_CONFIG_DIR}/libadda_cuda_backend_single.so"
    CACHE FILEPATH "Native Linux float32 CUDA runtime")
add_subdirectory("${_ADDA_ROOT}" "adda-core")
