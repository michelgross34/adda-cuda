cmake_minimum_required(VERSION 3.20)

project(ADDA LANGUAGES C)

# This target is the sequential equivalent of src/seq/Makefile.  The original
# CPU executable is always built; ADDA_CUDA adds a second executable with the
# CUDA MatVec backend and does not alter the CPU target.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Release or Debug)" FORCE)
endif()
option(ADDA_NO_FORTRAN "Disable optional Fortran routines" OFF)
option(ADDA_SPARSE "Build sparse (non-FFT) mode" OFF)
option(ADDA_PRECISE_TIMING "Enable precise timing" OFF)
option(INFO "Show detailed CUDA memory reservation and MatVec timing information" OFF)
option(ADDA_NOT_USE_LOCK "Disable file/system locking" OFF)
option(ADDA_ONLY_LOCKFILE "Use lock files only" OFF)
option(ADDA_OVERRIDE_STDC_TEST "Override the C99 conformance check" OFF)
option(ADDA_NO_GITHASH "Do not include the current Git revision" OFF)
option(ADDA_CUDA "Build double-precision CUDA targets" ON)
option(ADDA_CUDA_SPLIT_BACKEND
       "Use kernel.cu with the CMake-compiled C++ wrapper backend"
       ON)
option(ADDA_CUDA_SLICE "Build double-precision adda_cuda_slice" ON)
option(ADDA_CUDA_LOW_MEM "Build the preserved double-precision low-memory reference as adda_low_mem_double" ON)
option(ADDA_SINGLE "Build CPU single-data-path executable adda_single" ON)
option(ADDA_CUDA_SINGLE "Build full-grid float32 CUDA executable adda_cuda_single" ON)
option(ADDA_CUDA_SINGLE_SLICE "Build float32 slice executable adda_single_slice" ON)
option(ADDA_CUDA_LOW_MEM_SINGLE "Build float32 slice+first-octant executable adda_low_mem" ON)
# IMPORTANT: CMake never compiles .cu files.  In split mode the scripts build
# only kernel.cu with nvcc, while CMake enables GNU C++ and builds the host
# wrappermatvec_backend.cpp.  In historical mode the prebuilt monolithic
# cudamatvec_backend.cu library remains external to CMake.
if(CMAKE_BUILD_TYPE MATCHES "^[Dd]ebug$")
    set(_ADDA_CUDA_CONFIG_DIR "debug")
else()
    set(_ADDA_CUDA_CONFIG_DIR "release")
endif()
if(WIN32)
    set(_ADDA_CUDA_BACKEND_DEFAULT "${CMAKE_CURRENT_SOURCE_DIR}/cuda-backend/${_ADDA_CUDA_CONFIG_DIR}/libadda_cuda_backend.a")
    set(_ADDA_CUDA_RUNTIME_DEFAULT "${CMAKE_CURRENT_SOURCE_DIR}/cuda-backend/${_ADDA_CUDA_CONFIG_DIR}/adda_cuda_backend.dll")
    set(_ADDA_CUDA_SINGLE_BACKEND_DEFAULT "${CMAKE_CURRENT_SOURCE_DIR}/cuda-backend/${_ADDA_CUDA_CONFIG_DIR}/libadda_cuda_backend_single.a")
    set(_ADDA_CUDA_SINGLE_RUNTIME_DEFAULT "${CMAKE_CURRENT_SOURCE_DIR}/cuda-backend/${_ADDA_CUDA_CONFIG_DIR}/adda_cuda_backend_single.dll")
else()
    set(_ADDA_CUDA_BACKEND_DEFAULT "${CMAKE_CURRENT_SOURCE_DIR}/cuda-backend/${_ADDA_CUDA_CONFIG_DIR}/libadda_cuda_backend.so")
    set(_ADDA_CUDA_RUNTIME_DEFAULT "${_ADDA_CUDA_BACKEND_DEFAULT}")
    set(_ADDA_CUDA_SINGLE_BACKEND_DEFAULT "${CMAKE_CURRENT_SOURCE_DIR}/cuda-backend/${_ADDA_CUDA_CONFIG_DIR}/libadda_cuda_backend_single.so")
    set(_ADDA_CUDA_SINGLE_RUNTIME_DEFAULT "${_ADDA_CUDA_SINGLE_BACKEND_DEFAULT}")
endif()
set(ADDA_CUDA_EXTERNAL_BACKEND
    "${_ADDA_CUDA_BACKEND_DEFAULT}"
    CACHE FILEPATH "Link library for the separately-built CUDA backend")
set(ADDA_CUDA_EXTERNAL_DLL
    "${_ADDA_CUDA_RUNTIME_DEFAULT}"
    CACHE FILEPATH "Runtime library produced separately by nvcc")
set(ADDA_CUDA_SINGLE_EXTERNAL_BACKEND
    "${_ADDA_CUDA_SINGLE_BACKEND_DEFAULT}"
    CACHE FILEPATH "Link library for the separately-built float32 CUDA backend")
set(ADDA_CUDA_SINGLE_EXTERNAL_DLL
    "${_ADDA_CUDA_SINGLE_RUNTIME_DEFAULT}"
    CACHE FILEPATH "Float32 runtime library produced separately by nvcc")
if(ADDA_CUDA_EXTERNAL_BACKEND STREQUAL "")
    set(ADDA_CUDA_EXTERNAL_BACKEND
        "${_ADDA_CUDA_BACKEND_DEFAULT}"
        CACHE FILEPATH "Link library for the separately-built CUDA backend" FORCE)
endif()
if(ADDA_CUDA_EXTERNAL_DLL STREQUAL "")
    set(ADDA_CUDA_EXTERNAL_DLL
        "${_ADDA_CUDA_RUNTIME_DEFAULT}"
        CACHE FILEPATH "Runtime library produced separately by nvcc" FORCE)
endif()
if(ADDA_CUDA_SINGLE_EXTERNAL_BACKEND STREQUAL "")
    set(ADDA_CUDA_SINGLE_EXTERNAL_BACKEND
        "${_ADDA_CUDA_SINGLE_BACKEND_DEFAULT}"
        CACHE FILEPATH "Link library for the separately-built float32 CUDA backend" FORCE)
endif()
if(ADDA_CUDA_SINGLE_EXTERNAL_DLL STREQUAL "")
    set(ADDA_CUDA_SINGLE_EXTERNAL_DLL
        "${_ADDA_CUDA_SINGLE_RUNTIME_DEFAULT}"
        CACHE FILEPATH "Float32 runtime library produced separately by nvcc" FORCE)
endif()

if((ADDA_CUDA OR ADDA_CUDA_SINGLE OR ADDA_CUDA_SINGLE_SLICE OR ADDA_CUDA_LOW_MEM_SINGLE) AND ADDA_SPARSE)
    message(FATAL_ERROR "CUDA MatVec targets require FFT mode and cannot be built with ADDA_SPARSE.")
endif()

if(WIN32 AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/fftw/fftw3.h")
    set(_ADDA_DEFAULT_FFTW3_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/fftw")
elseif(WIN32 AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/fftw3/fftw3.h")
    set(_ADDA_DEFAULT_FFTW3_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/fftw3")
else()
    set(_ADDA_DEFAULT_FFTW3_ROOT "")
endif()
set(FFTW3_ROOT "${_ADDA_DEFAULT_FFTW3_ROOT}" CACHE PATH "Root directory of an FFTW3 installation")
set(FFTW3_INCLUDE_DIR "" CACHE PATH "Directory containing fftw3.h")
set(FFTW3_LIBRARY "" CACHE FILEPATH "FFTW3 double library or import library")
set(FFTW3F_LIBRARY "" CACHE FILEPATH "FFTW3 float library or import library")

set(ADDA_C_SOURCES
    src/ADDAmain.c
    src/CalculateE.c
    src/calculator.c
    src/chebyshev.c
    src/cmplx.c
    src/comm.c
    src/crosssec.c
    src/GenerateB.c
    src/interaction.c
    src/io.c
    src/iterative.c
    src/linalg.c
    src/make_particle.c
    src/matvec.c
    src/memory.c
    src/mt19937ar.c
    src/param.c
    src/Romberg.c
    src/sinint.c
    src/somnec.c
    src/timing.c
    src/vars.c
    src/igt_so.c
)

if(ADDA_PRECISE_TIMING)
    add_compile_definitions(PRECISE_TIMING)
    list(APPEND ADDA_C_SOURCES src/prec_time.c)
endif()

if(ADDA_NO_FORTRAN)
    add_compile_definitions(NO_FORTRAN)
else()
    include(CheckLanguage)
    check_language(Fortran)
    if(NOT CMAKE_Fortran_COMPILER)
        message(FATAL_ERROR "ADDA_NO_FORTRAN=OFF requires gfortran (or another Fortran compiler).")
    endif()
    enable_language(Fortran)
    list(APPEND ADDA_C_SOURCES
        src/fort/d07hre.f
        src/fort/d09hre.f
        src/fort/d113re.f
        src/fort/d132re.f
        src/fort/dadhre.f
        src/fort/dchhre.f
        src/fort/dcuhre.f
        src/fort/dfshre.f
        src/fort/dinhre.f
        src/fort/drlhre.f
        src/fort/dtrhre.f
        src/fort/propaesplibreintadda.f
        src/fort/bessel.f90
    )
endif()

if(ADDA_SPARSE)
    add_compile_definitions(SPARSE)
else()
    # fft.h selects FFTW3 by default, just as src/Makefile does.  CUDA still
    # needs FFTW3 during one-time D/R Green-tensor construction; only MatVec
    # itself switches to cuFFT.
    list(APPEND ADDA_C_SOURCES src/fft.c)
    if(FFTW3_INCLUDE_DIR)
        set(FFTW3_INCLUDE_DIR_FOUND "${FFTW3_INCLUDE_DIR}")
    else()
        find_path(FFTW3_INCLUDE_DIR_FOUND fftw3.h
            HINTS "${FFTW3_ROOT}" "${FFTW3_ROOT}/include")
    endif()
    if(FFTW3_LIBRARY)
        set(FFTW3_LIBRARY_FOUND "${FFTW3_LIBRARY}")
    elseif(MINGW AND EXISTS "${FFTW3_ROOT}/libfftw3.dll.a")
        set(FFTW3_LIBRARY_FOUND "${FFTW3_ROOT}/libfftw3.dll.a")
    else()
        find_library(FFTW3_LIBRARY_FOUND NAMES fftw3 fftw3-3 libfftw3-3
            HINTS "${FFTW3_ROOT}" "${FFTW3_ROOT}/lib" "${FFTW3_ROOT}/bin")
    endif()
    if(NOT FFTW3_INCLUDE_DIR_FOUND OR NOT FFTW3_LIBRARY_FOUND)
        message(FATAL_ERROR
            "FFTW3 double precision was not found. Set FFTW3_ROOT (or FFTW3_INCLUDE_DIR and FFTW3_LIBRARY), "
            "or configure with -DADDA_SPARSE=ON.")
    endif()
    set(FFTW3_INCLUDE_DIR "${FFTW3_INCLUDE_DIR_FOUND}" CACHE PATH "Directory containing fftw3.h" FORCE)
    set(FFTW3_LIBRARY "${FFTW3_LIBRARY_FOUND}" CACHE FILEPATH "FFTW3 double library or import library" FORCE)
    if(WIN32)
        find_file(FFTW3_RUNTIME_DLL libfftw3-3.dll
            HINTS "${FFTW3_ROOT}" "${FFTW3_ROOT}/bin")
    endif()

    if(ADDA_SINGLE OR ADDA_CUDA_SINGLE OR ADDA_CUDA_SINGLE_SLICE OR ADDA_CUDA_LOW_MEM_SINGLE)
        if(FFTW3F_LIBRARY)
            set(FFTW3F_LIBRARY_FOUND "${FFTW3F_LIBRARY}")
        elseif(MINGW AND EXISTS "${FFTW3_ROOT}/libfftw3f.dll.a")
            set(FFTW3F_LIBRARY_FOUND "${FFTW3_ROOT}/libfftw3f.dll.a")
        else()
            find_library(FFTW3F_LIBRARY_FOUND NAMES fftw3f fftw3f-3 libfftw3f-3
                HINTS "${FFTW3_ROOT}" "${FFTW3_ROOT}/lib" "${FFTW3_ROOT}/bin")
        endif()
        if(NOT FFTW3F_LIBRARY_FOUND)
            message(FATAL_ERROR
                "FFTW3 single precision was not found. Install/provide libfftw3f-3 and set FFTW3F_LIBRARY.")
        endif()
        set(FFTW3F_LIBRARY "${FFTW3F_LIBRARY_FOUND}" CACHE FILEPATH "FFTW3 float library or import library" FORCE)
        if(WIN32)
            find_file(FFTW3F_RUNTIME_DLL libfftw3f-3.dll
                HINTS "${FFTW3_ROOT}" "${FFTW3_ROOT}/bin")
        endif()
    endif()
endif()

if(ADDA_NOT_USE_LOCK)
    add_compile_definitions(NOT_USE_LOCK)
elseif(ADDA_ONLY_LOCKFILE)
    add_compile_definitions(ONLY_LOCKFILE)
endif()
if(ADDA_OVERRIDE_STDC_TEST)
    add_compile_definitions(OVERRIDE_STDC_TEST)
endif()

if(ADDA_NO_GITHASH)
    add_compile_definitions(NO_GITHASH)
else()
    execute_process(
        COMMAND git log --pretty=format:%h -n 1
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE ADDA_GITHASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE ADDA_GIT_RESULT)
    if(NOT ADDA_GIT_RESULT EQUAL 0)
        set(ADDA_GITHASH "")
    endif()
    if(ADDA_GITHASH)
        set(ADDA_GITHASH_DEFINE "#define GITHASH \"${ADDA_GITHASH}\"")
    else()
        set(ADDA_GITHASH_DEFINE "")
    endif()
    configure_file(src/githash.h.in "${CMAKE_CURRENT_BINARY_DIR}/githash.h" @ONLY)
endif()

function(configure_adda_target target precision)
    target_include_directories(${target} PRIVATE src)
    if(NOT ADDA_NO_GITHASH)
        target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    endif()
    if(NOT ADDA_SPARSE)
        target_include_directories(${target} PRIVATE "${FFTW3_INCLUDE_DIR}")
        if(precision STREQUAL "single")
            target_link_libraries(${target} PRIVATE "${FFTW3F_LIBRARY}")
            target_compile_definitions(${target} PRIVATE ADDA_SINGLE)
            if(FFTW3F_RUNTIME_DLL)
                add_custom_command(TARGET ${target} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${FFTW3F_RUNTIME_DLL}"
                        "$<TARGET_FILE_DIR:${target}>/libfftw3f-3.dll")
            endif()
        else()
            target_link_libraries(${target} PRIVATE "${FFTW3_LIBRARY}")
            if(FFTW3_RUNTIME_DLL)
                add_custom_command(TARGET ${target} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${FFTW3_RUNTIME_DLL}"
                        "$<TARGET_FILE_DIR:${target}>/libfftw3-3.dll")
            endif()
        endif()
    elseif(precision STREQUAL "single")
        target_compile_definitions(${target} PRIVATE ADDA_SINGLE)
    endif()
    if(NOT MSVC)
        target_link_libraries(${target} PRIVATE m)
    endif()

    set_target_properties(${target} PROPERTIES
        C_STANDARD 99
        C_STANDARD_REQUIRED YES
        C_EXTENSIONS NO
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE
            $<$<AND:$<COMPILE_LANGUAGE:C>,$<CONFIG:Release>>:-Ofast;-w>
            $<$<AND:$<COMPILE_LANGUAGE:C>,$<CONFIG:Debug>>:-Og;-g;-Wall;-Wextra;-Wcast-qual;-Wpointer-arith;-Wwrite-strings;-Wstrict-prototypes;-Wformat=2;-Wshadow;-Wcast-align;-Wcomment>)
    endif()
endfunction()

add_executable(adda ${ADDA_C_SOURCES})
configure_adda_target(adda double)

if(ADDA_SINGLE)
    add_executable(adda_single ${ADDA_C_SOURCES})
    configure_adda_target(adda_single single)
endif()

if(ADDA_CUDA OR ADDA_CUDA_SINGLE OR ADDA_CUDA_SINGLE_SLICE OR ADDA_CUDA_LOW_MEM_SINGLE)
    if(ADDA_CUDA_SPLIT_BACKEND)
        # Split backend: only kernel.cu is compiled by nvcc (outside CMake).
        # All host-side CUDA orchestration stays in wrappermatvec_backend.cpp,
        # which is a normal GNU C++ translation unit and is debuggable in CLion.
        enable_language(CXX)
        if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            message(FATAL_ERROR
                "ADDA_CUDA_SPLIT_BACKEND requires GNU g++ (MinGW g++ on Windows, g++ on Linux/WSL). "
                "Detected C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER}")
        endif()
        set(CMAKE_CXX_STANDARD 14)
        set(CMAKE_CXX_STANDARD_REQUIRED ON)
        find_package(CUDAToolkit REQUIRED)

        set(_ADDA_KERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cuda-backend/kernels")
        if(WIN32)
            set(_ADDA_KERNEL_DOUBLE_LINK "${_ADDA_KERNEL_DIR}/libadda_cuda_kernels.a")
            set(_ADDA_KERNEL_DOUBLE_RUNTIME "${_ADDA_KERNEL_DIR}/adda_cuda_kernels.dll")
            set(_ADDA_KERNEL_SINGLE_LINK "${_ADDA_KERNEL_DIR}/libadda_cuda_kernels_single.a")
            set(_ADDA_KERNEL_SINGLE_RUNTIME "${_ADDA_KERNEL_DIR}/adda_cuda_kernels_single.dll")
        else()
            set(_ADDA_KERNEL_DOUBLE_LINK "${_ADDA_KERNEL_DIR}/libadda_cuda_kernels.so")
            set(_ADDA_KERNEL_DOUBLE_RUNTIME "${_ADDA_KERNEL_DOUBLE_LINK}")
            set(_ADDA_KERNEL_SINGLE_LINK "${_ADDA_KERNEL_DIR}/libadda_cuda_kernels_single.so")
            set(_ADDA_KERNEL_SINGLE_RUNTIME "${_ADDA_KERNEL_SINGLE_LINK}")
        endif()
        foreach(_ADDA_REQUIRED_KERNEL IN ITEMS
                "${_ADDA_KERNEL_DOUBLE_LINK}" "${_ADDA_KERNEL_DOUBLE_RUNTIME}"
                "${_ADDA_KERNEL_SINGLE_LINK}" "${_ADDA_KERNEL_SINGLE_RUNTIME}")
            if(NOT EXISTS "${_ADDA_REQUIRED_KERNEL}")
                message(FATAL_ERROR
                    "Split CUDA kernel library not found: ${_ADDA_REQUIRED_KERNEL}. "
                    "Run scripts/build_cuda_backends.bat (Windows) or scripts/build_cuda_backends.sh (Linux/WSL) first.")
            endif()
        endforeach()

        if(WIN32)
            add_library(adda_cuda_kernels SHARED IMPORTED GLOBAL)
            set_target_properties(adda_cuda_kernels PROPERTIES
                IMPORTED_IMPLIB "${_ADDA_KERNEL_DOUBLE_LINK}"
                IMPORTED_LOCATION "${_ADDA_KERNEL_DOUBLE_RUNTIME}")
            add_library(adda_cuda_kernels_single SHARED IMPORTED GLOBAL)
            set_target_properties(adda_cuda_kernels_single PROPERTIES
                IMPORTED_IMPLIB "${_ADDA_KERNEL_SINGLE_LINK}"
                IMPORTED_LOCATION "${_ADDA_KERNEL_SINGLE_RUNTIME}")
        else()
            add_library(adda_cuda_kernels SHARED IMPORTED GLOBAL)
            set_target_properties(adda_cuda_kernels PROPERTIES IMPORTED_LOCATION "${_ADDA_KERNEL_DOUBLE_LINK}")
            add_library(adda_cuda_kernels_single SHARED IMPORTED GLOBAL)
            set_target_properties(adda_cuda_kernels_single PROPERTIES IMPORTED_LOCATION "${_ADDA_KERNEL_SINGLE_LINK}")
        endif()

        add_library(adda_cuda_backend SHARED src/wrappermatvec_backend.cpp)
        add_library(adda_cuda_backend_single SHARED src/wrappermatvec_backend.cpp)
        target_compile_definitions(adda_cuda_backend_single PRIVATE ADDA_CUDA_SINGLE_BACKEND)
        target_include_directories(adda_cuda_backend PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
        target_include_directories(adda_cuda_backend_single PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
        target_link_libraries(adda_cuda_backend PRIVATE
            adda_cuda_kernels CUDA::cudart CUDA::cufft CUDA::cublas)
        target_link_libraries(adda_cuda_backend_single PRIVATE
            adda_cuda_kernels_single CUDA::cudart CUDA::cufft CUDA::cublas)
        if(WIN32)
            target_sources(adda_cuda_backend PRIVATE src/cudamatvec_backend.def)
            target_sources(adda_cuda_backend_single PRIVATE src/cudamatvec_backend_single.def)
        else()
            set_target_properties(adda_cuda_backend adda_cuda_backend_single PROPERTIES BUILD_RPATH "$ORIGIN")
        endif()
        set_target_properties(adda_cuda_backend PROPERTIES
            OUTPUT_NAME "adda_cuda_backend"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
            ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
        set_target_properties(adda_cuda_backend_single PROPERTIES
            OUTPUT_NAME "adda_cuda_backend_single"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
            ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

        # The kernel DLL/SO is not configuration-specific.  Put it next to the
        # CMake-built wrapper so the runtime loader sees the same dependency in
        # Debug and Release.
        add_custom_command(TARGET adda_cuda_backend POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_ADDA_KERNEL_DOUBLE_RUNTIME}" "$<TARGET_FILE_DIR:adda_cuda_backend>")
        add_custom_command(TARGET adda_cuda_backend_single POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_ADDA_KERNEL_SINGLE_RUNTIME}" "$<TARGET_FILE_DIR:adda_cuda_backend_single>")
    else()
        # Historical monolithic backend.  src/cudamatvec_backend.cu remains
        # untouched and the existing prebuilt Debug/Release libraries are used.
        if(ADDA_CUDA AND NOT EXISTS "${ADDA_CUDA_EXTERNAL_BACKEND}")
            message(FATAL_ERROR
                "CUDA double backend not found: ${ADDA_CUDA_EXTERNAL_BACKEND}. "
                "ADDA_CUDA_SPLIT_BACKEND=OFF requires the historical prebuilt backend.")
        endif()
        if((ADDA_CUDA_SINGLE OR ADDA_CUDA_SINGLE_SLICE OR ADDA_CUDA_LOW_MEM_SINGLE)
           AND NOT EXISTS "${ADDA_CUDA_SINGLE_EXTERNAL_BACKEND}")
            message(FATAL_ERROR
                "CUDA float32 backend not found: ${ADDA_CUDA_SINGLE_EXTERNAL_BACKEND}. "
                "ADDA_CUDA_SPLIT_BACKEND=OFF requires the historical prebuilt backend.")
        endif()
        if(WIN32 AND ADDA_CUDA AND NOT EXISTS "${ADDA_CUDA_EXTERNAL_DLL}")
            message(FATAL_ERROR "CUDA double runtime DLL not found: ${ADDA_CUDA_EXTERNAL_DLL}")
        endif()
        if(WIN32 AND (ADDA_CUDA_SINGLE OR ADDA_CUDA_SINGLE_SLICE OR ADDA_CUDA_LOW_MEM_SINGLE)
           AND NOT EXISTS "${ADDA_CUDA_SINGLE_EXTERNAL_DLL}")
            message(FATAL_ERROR "CUDA float32 runtime DLL not found: ${ADDA_CUDA_SINGLE_EXTERNAL_DLL}")
        endif()
        if(WIN32)
            add_library(adda_cuda_backend SHARED IMPORTED GLOBAL)
            set_target_properties(adda_cuda_backend PROPERTIES
                IMPORTED_IMPLIB "${ADDA_CUDA_EXTERNAL_BACKEND}"
                IMPORTED_LOCATION "${ADDA_CUDA_EXTERNAL_DLL}")
            add_library(adda_cuda_backend_single SHARED IMPORTED GLOBAL)
            set_target_properties(adda_cuda_backend_single PROPERTIES
                IMPORTED_IMPLIB "${ADDA_CUDA_SINGLE_EXTERNAL_BACKEND}"
                IMPORTED_LOCATION "${ADDA_CUDA_SINGLE_EXTERNAL_DLL}")
        else()
            add_library(adda_cuda_backend UNKNOWN IMPORTED GLOBAL)
            set_target_properties(adda_cuda_backend PROPERTIES IMPORTED_LOCATION "${ADDA_CUDA_EXTERNAL_BACKEND}")
            add_library(adda_cuda_backend_single UNKNOWN IMPORTED GLOBAL)
            set_target_properties(adda_cuda_backend_single PROPERTIES IMPORTED_LOCATION "${ADDA_CUDA_SINGLE_EXTERNAL_BACKEND}")
        endif()
    endif()

    set(ADDA_CUDA_SOURCES ${ADDA_C_SOURCES})
    list(REMOVE_ITEM ADDA_CUDA_SOURCES src/matvec.c)
    list(APPEND ADDA_CUDA_SOURCES src/cudamatvec.c src/lanier_precon.c src/lanier_full_precon.c)

    if(ADDA_CUDA)
        add_executable(adda_cuda ${ADDA_CUDA_SOURCES})
        target_compile_definitions(adda_cuda PRIVATE ADDA_CUDA)
        target_link_libraries(adda_cuda PRIVATE adda_cuda_backend)
        configure_adda_target(adda_cuda double)

        if(ADDA_CUDA_SLICE)
            add_executable(adda_cuda_slice ${ADDA_CUDA_SOURCES})
            target_compile_definitions(adda_cuda_slice PRIVATE ADDA_CUDA ADDA_CUDA_SLICE)
            target_link_libraries(adda_cuda_slice PRIVATE adda_cuda_backend)
            configure_adda_target(adda_cuda_slice double)
        endif()

        if(ADDA_CUDA_LOW_MEM)
            add_executable(adda_low_mem_double ${ADDA_CUDA_SOURCES})
            target_compile_definitions(adda_low_mem_double PRIVATE ADDA_CUDA ADDA_CUDA_SLICE ADDA_CUDA_LOW_MEM)
            target_link_libraries(adda_low_mem_double PRIVATE adda_cuda_backend)
            configure_adda_target(adda_low_mem_double double)
        endif()
    endif()

    if(ADDA_CUDA_SINGLE)
        add_executable(adda_cuda_single ${ADDA_CUDA_SOURCES})
        target_compile_definitions(adda_cuda_single PRIVATE ADDA_CUDA)
        target_link_libraries(adda_cuda_single PRIVATE adda_cuda_backend_single)
        configure_adda_target(adda_cuda_single single)
    endif()

    if(ADDA_CUDA_SINGLE_SLICE)
        add_executable(adda_single_slice ${ADDA_CUDA_SOURCES})
        target_compile_definitions(adda_single_slice PRIVATE ADDA_CUDA ADDA_CUDA_SLICE)
        target_link_libraries(adda_single_slice PRIVATE adda_cuda_backend_single)
        configure_adda_target(adda_single_slice single)
    endif()

    if(ADDA_CUDA_LOW_MEM_SINGLE)
        add_executable(adda_low_mem ${ADDA_CUDA_SOURCES})
        target_compile_definitions(adda_low_mem PRIVATE ADDA_CUDA ADDA_CUDA_SLICE ADDA_CUDA_LOW_MEM)
        target_link_libraries(adda_low_mem PRIVATE adda_cuda_backend_single)
        configure_adda_target(adda_low_mem single)
    endif()

    if(WIN32 AND NOT ADDA_CUDA_SPLIT_BACKEND)
        foreach(tgt IN ITEMS adda_cuda adda_cuda_slice adda_low_mem_double)
            if(TARGET ${tgt})
                add_custom_command(TARGET ${tgt} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${ADDA_CUDA_EXTERNAL_DLL}" "$<TARGET_FILE_DIR:${tgt}>")
            endif()
        endforeach()
        foreach(tgt IN ITEMS adda_cuda_single adda_single_slice adda_low_mem)
            if(TARGET ${tgt})
                add_custom_command(TARGET ${tgt} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${ADDA_CUDA_SINGLE_EXTERNAL_DLL}" "$<TARGET_FILE_DIR:${tgt}>")
            endif()
        endforeach()
    endif()

    if(INFO)
        foreach(tgt IN ITEMS adda_cuda adda_cuda_slice adda_low_mem_double adda_cuda_single adda_single_slice adda_low_mem)
            if(TARGET ${tgt})
                target_compile_definitions(${tgt} PRIVATE ADDA_CUDA_INFO)
            endif()
        endforeach()
    endif()
    message(STATUS "CUDA detailed INFO output: ${INFO}")
    message(STATUS "ADDA_CUDA_SPLIT_BACKEND: ${ADDA_CUDA_SPLIT_BACKEND}")
    if(ADDA_CUDA_SPLIT_BACKEND)
        message(STATUS "CUDA host wrapper is compiled by GNU C++: ${CMAKE_CXX_COMPILER}")
        message(STATUS "CUDA kernels: ${_ADDA_KERNEL_DIR}")
    else()
        message(STATUS "Historical CUDA backend: ${ADDA_CUDA_EXTERNAL_BACKEND}")
        message(STATUS "Historical float32 backend: ${ADDA_CUDA_SINGLE_EXTERNAL_BACKEND}")
    endif()
endif()

# LANIER_FULL_ALL_CUDA_ALL_SOLVERS_V1
# Keep the actor beside the executables so LANIER_FULL also works when ADDA is
# launched directly from <build>\\bin. ADDA_LANIER_ACTOR still overrides this.
set(ADDA_LANIER_ACTOR_FILE "${CMAKE_CURRENT_SOURCE_DIR}/ai/lanier_tqc_v1_ema_actor.bin")
if(EXISTS "${ADDA_LANIER_ACTOR_FILE}")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/bin/ai")
    configure_file("${ADDA_LANIER_ACTOR_FILE}"
                   "${CMAKE_BINARY_DIR}/bin/ai/lanier_tqc_v1_ema_actor.bin" COPYONLY)
else()
    message(WARNING "LANIER_FULL actor not found: ${ADDA_LANIER_ACTOR_FILE}")
endif()
# END_LANIER_FULL_ALL_CUDA_ALL_SOLVERS_V1
