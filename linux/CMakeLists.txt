cmake_minimum_required(VERSION 3.20)

# Linux / WSL entry point. The implementation and source lists remain in the
# repository-root CMakeLists.txt; this file only selects the Linux/WSL
# configuration.
project(ADDA_LINUX LANGUAGES C)

if(WIN32)
    message(FATAL_ERROR "linux/CMakeLists.txt must be configured on Linux/WSL.")
endif()

# On Linux/WSL the complete CUDA backend is built separately by
# scripts/build_cuda_backends.sh.
#
# FORCE is intentional so an old CLion/CMake cache cannot leave
# ADDA_CUDA_SPLIT_BACKEND=ON.
set(ADDA_CUDA_SPLIT_BACKEND OFF CACHE BOOL
    "Use prebuilt complete CUDA backend on Linux/WSL"
    FORCE)

set(ADDA_CUDA ON CACHE BOOL "Build double-precision CUDA targets" FORCE)
set(ADDA_CUDA_SLICE ON CACHE BOOL "Build double-precision CUDA slice target" FORCE)
set(ADDA_CUDA_LOW_MEM ON CACHE BOOL "Build double-precision low-memory CUDA target" FORCE)
set(ADDA_SINGLE ON CACHE BOOL "Build CPU single-precision target" FORCE)
set(ADDA_CUDA_SINGLE ON CACHE BOOL "Build float32 CUDA target" FORCE)
set(ADDA_CUDA_SINGLE_SLICE ON CACHE BOOL "Build float32 CUDA slice target" FORCE)
set(ADDA_CUDA_LOW_MEM_SINGLE ON CACHE BOOL "Build float32 low-memory CUDA target" FORCE)

set(_ADDA_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")

# Select the prebuilt Linux CUDA backend matching the active CMake build type.
# scripts/build_cuda_backends.sh builds both debug and release backends.
if(CMAKE_BUILD_TYPE MATCHES "^[Dd]ebug$")
    set(_ADDA_CUDA_CONFIG_DIR "debug")
else()
    set(_ADDA_CUDA_CONFIG_DIR "release")
endif()

set(ADDA_CUDA_EXTERNAL_BACKEND
    "${_ADDA_ROOT}/cuda-backend/${_ADDA_CUDA_CONFIG_DIR}/libadda_cuda_backend.so"
    CACHE FILEPATH "Linux double CUDA backend" FORCE)

set(ADDA_CUDA_EXTERNAL_DLL
    "${ADDA_CUDA_EXTERNAL_BACKEND}"
    CACHE FILEPATH "Linux double CUDA runtime library" FORCE)

set(ADDA_CUDA_SINGLE_EXTERNAL_BACKEND
    "${_ADDA_ROOT}/cuda-backend/${_ADDA_CUDA_CONFIG_DIR}/libadda_cuda_backend_single.so"
    CACHE FILEPATH "Linux float32 CUDA backend" FORCE)

set(ADDA_CUDA_SINGLE_EXTERNAL_DLL
    "${ADDA_CUDA_SINGLE_EXTERNAL_BACKEND}"
    CACHE FILEPATH "Linux float32 CUDA runtime library" FORCE)

add_subdirectory("${_ADDA_ROOT}" "adda-core")

# Keep every executable in the bin directory of the active CLion/CMake profile.
set(_ADDA_LINUX_BIN_DIR "${CMAKE_BINARY_DIR}/bin")
foreach(_ADDA_TARGET
    adda
    adda_single
    adda_cuda
    adda_cuda_slice
    adda_low_mem_double
    adda_cuda_single
    adda_single_slice
    adda_low_mem)
    if(TARGET ${_ADDA_TARGET})
        set_target_properties(${_ADDA_TARGET} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${_ADDA_LINUX_BIN_DIR}")
    endif()
endforeach()
