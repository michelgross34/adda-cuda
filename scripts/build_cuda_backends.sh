#!/usr/bin/env bash
set -euo pipefail

# Build the split CUDA kernel libraries.  kernel.cu is the only source compiled
# by nvcc; wrappermatvec_backend.cpp is compiled later by CMake with GNU g++.
# There is deliberately no Debug-specific kernel binary.  The same kernel .so
# is used by CMake Debug and Release profiles.
# Usage: scripts/build_cuda_backends.sh [CUDA_ARCH]

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(CDPATH= cd -- "${script_dir}/.." && pwd)
output_dir="${project_root}/cuda-backend/kernels"
source_file="${project_root}/src/kernel.cu"
include_dir="${project_root}/src"

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    printf 'Usage: %s [native|all|all-major|SM]\n' "$0"
    exit 0
fi
if [[ $(uname -s) != "Linux" ]]; then
    printf 'ERROR: this script builds Linux/WSL kernel .so libraries and must run on Linux.\n' >&2
    exit 1
fi
if (( $# > 1 )); then
    printf 'ERROR: expected at most one CUDA architecture argument.\n' >&2
    exit 2
fi

cuda_arch=${1:-native}
cuda_arch=${cuda_arch#sm_}
case "${cuda_arch}" in
    native|all|all-major) arch_flag="-arch=${cuda_arch}" ;;
    *[!0-9]*|'')
        printf 'ERROR: invalid CUDA architecture: %s\n' "${cuda_arch}" >&2
        exit 2 ;;
    *) arch_flag="-arch=sm_${cuda_arch}" ;;
esac

command -v nvcc >/dev/null 2>&1 || { printf 'ERROR: nvcc was not found in PATH.\n' >&2; exit 1; }
[[ -f "${source_file}" ]] || { printf 'ERROR: CUDA kernel source not found: %s\n' "${source_file}" >&2; exit 1; }
[[ -f "${include_dir}/kernel.h" ]] || { printf 'ERROR: CUDA kernel header not found.\n' >&2; exit 1; }
mkdir -p "${output_dir}"

build_one() {
    local label=$1
    local define=$2
    local output=$3
    printf '[CUDA kernels %s] nvcc -> %s (%s)\n' "${label}" "${output}" "${arch_flag}"
    nvcc -std=c++14 --shared -O3 "${arch_flag}" -Xcompiler=-fPIC \
        ${define} -I"${include_dir}" "${source_file}" \
        -Xlinker=-soname -Xlinker="$(basename "${output}")" \
        -o "${output}"
    test -s "${output}"
}

build_one double "" "${output_dir}/libadda_cuda_kernels.so"
build_one float32 "-DADDA_CUDA_SINGLE_BACKEND" "${output_dir}/libadda_cuda_kernels_single.so"

printf '\nSplit CUDA kernel libraries built successfully in %s.\n' "${output_dir}"
printf 'CMake Debug and Release both use these same kernel libraries; host debugging is in wrappermatvec_backend.cpp.\n'
