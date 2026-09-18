#!/usr/bin/env bash
set -euo pipefail

# Build every Linux executable and collect runtime files in build_linux/bin.
#
# Linux/WSL always uses the monolithic CUDA backend:
#   ADDA_CUDA_SPLIT_BACKEND=OFF
#
# scripts/build_cuda_backends.sh builds the complete CUDA backends outside CMake:
#
#   cuda-backend/debug/libadda_cuda_backend.so
#   cuda-backend/debug/libadda_cuda_backend_single.so
#   cuda-backend/release/libadda_cuda_backend.so
#   cuda-backend/release/libadda_cuda_backend_single.so
#
# CMake does NOT compile CUDA sources and does NOT use split kernel libraries
# on Linux/WSL.  The Release monolithic .so files are copied next to the
# executables in build_linux/bin.
#
# Usage: scripts/build_all.sh [CUDA_ARCH] [JOBS]
# Example: scripts/build_all.sh 89 8

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    printf 'Usage: %s [native|all|all-major|SM] [JOBS]\n' "$0"
    exit 0
fi

if [[ $(uname -s) != "Linux" ]]; then
    printf 'ERROR: this script must run on Linux.\n' >&2
    exit 1
fi

if (( $# > 2 )); then
    printf 'ERROR: expected at most CUDA_ARCH and JOBS arguments.\n' >&2
    exit 2
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(CDPATH= cd -- "${script_dir}/.." && pwd)
build_dir="${project_root}/build_linux"
bin_dir="${build_dir}/bin"
backend_dir="${project_root}/cuda-backend"
cuda_arch=${1:-native}
jobs=${2:-${ADDA_BUILD_JOBS:-}}

if [[ -z ${jobs} ]]; then
    if command -v nproc >/dev/null 2>&1; then
        jobs=$(nproc)
    elif command -v getconf >/dev/null 2>&1; then
        jobs=$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1')
    else
        jobs=1
    fi
fi

case "${jobs}" in
    *[!0-9]*|'')
        printf 'ERROR: JOBS must be a positive integer.\n' >&2
        exit 2
        ;;
esac

if (( jobs < 1 )); then
    printf 'ERROR: JOBS must be at least 1.\n' >&2
    exit 2
fi

if ! command -v cmake >/dev/null 2>&1; then
    printf 'ERROR: cmake was not found in PATH.\n' >&2
    exit 1
fi

if ! command -v gfortran >/dev/null 2>&1; then
    printf 'ERROR: gfortran was not found in PATH. Fortran is enabled by default.\n' >&2
    exit 1
fi

fortran_compiler=$(command -v gfortran)

if [[ ! -f "${script_dir}/build_cuda_backends.sh" ]]; then
    printf 'ERROR: CUDA build script not found: %s\n' \
        "${script_dir}/build_cuda_backends.sh" >&2
    exit 1
fi

printf '[setup] Parallel jobs: %s\n' "${jobs}"
printf '[1/4] Building monolithic CUDA libraries\n'
bash "${script_dir}/build_cuda_backends.sh" "${cuda_arch}"

# Linux/WSL uses ADDA_CUDA_SPLIT_BACKEND=OFF.
# Only the complete monolithic CUDA backends are required.
cuda_libraries=(
    "${backend_dir}/release/libadda_cuda_backend.so"
    "${backend_dir}/release/libadda_cuda_backend_single.so"
)

for library in "${cuda_libraries[@]}"; do
    if [[ ! -s ${library} ]]; then
        printf 'ERROR: required CUDA library not found: %s\n' "${library}" >&2
        exit 1
    fi
done

printf '[2/4] Configuring CMake in %s\n' "${build_dir}"
cmake -S "${project_root}/linux" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DADDA_NO_FORTRAN=OFF \
    -DCMAKE_Fortran_COMPILER="${fortran_compiler}" \
    -DADDA_LINUX_CUDA=ON \
    -DADDA_CUDA_SPLIT_BACKEND=OFF \
    '-DCMAKE_BUILD_RPATH=$ORIGIN'

printf '[3/4] Building all eight executables\n'
cmake --build "${build_dir}" --config Release --parallel "${jobs}" --target \
    adda adda_single adda_cuda adda_cuda_slice adda_low_mem_double \
    adda_cuda_single adda_single_slice adda_low_mem

printf '[4/4] Collecting shared libraries and AI resources\n'
mkdir -p "${bin_dir}/ai"

# Copy the complete monolithic CUDA runtime libraries next to the executables.
for library in \
    "${backend_dir}/release/libadda_cuda_backend.so" \
    "${backend_dir}/release/libadda_cuda_backend_single.so"
do
    if [[ ! -s ${library} ]]; then
        printf 'ERROR: required shared library not found: %s\n' "${library}" >&2
        exit 1
    fi
    cp -f "${library}" "${bin_dir}/"
done

if [[ ! -s "${project_root}/ai/lanier_tqc_v1_ema_actor.bin" ]]; then
    printf 'ERROR: AI resource not found: %s\n' \
        "${project_root}/ai/lanier_tqc_v1_ema_actor.bin" >&2
    exit 1
fi

cp -a "${project_root}/ai/." "${bin_dir}/ai/"

printf '[verify] Checking executables and monolithic CUDA runtime libraries\n'

executables=(
    adda
    adda_single
    adda_cuda
    adda_cuda_slice
    adda_low_mem_double
    adda_cuda_single
    adda_single_slice
    adda_low_mem
)

for executable in "${executables[@]}"; do
    if [[ ! -x "${bin_dir}/${executable}" ]]; then
        printf 'ERROR: executable missing: %s\n' "${bin_dir}/${executable}" >&2
        exit 1
    fi
done

runtime_libraries=(
    "${bin_dir}/libadda_cuda_backend.so"
    "${bin_dir}/libadda_cuda_backend_single.so"
)

for library in "${runtime_libraries[@]}"; do
    if [[ ! -s ${library} ]]; then
        printf 'ERROR: runtime CUDA library missing: %s\n' "${library}" >&2
        exit 1
    fi
done

printf '\nComplete Linux build available in: %s\n' "${bin_dir}"
printf '  %s executables\n' "${#executables[@]}"
printf '  CUDA backend mode: ADDA_CUDA_SPLIT_BACKEND=OFF\n'
printf '  Monolithic CUDA runtime libraries: %s\n' "${bin_dir}"
printf '  AI resources: %s\n' "${bin_dir}/ai"
