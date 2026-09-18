#!/usr/bin/env bash
set -euo pipefail

# Build the complete CUDA backend shared libraries outside CMake.
#
# Linux/WSL uses ADDA_CUDA_SPLIT_BACKEND=OFF.
# This script builds BOTH Debug and Release monolithic backends:
#
#   cuda-backend/debug/libadda_cuda_backend.so
#   cuda-backend/debug/libadda_cuda_backend_single.so
#   cuda-backend/release/libadda_cuda_backend.so
#   cuda-backend/release/libadda_cuda_backend_single.so
#
# No split kernel libraries are built on Linux/WSL.
#
# Usage:
#   scripts/build_cuda_backends.sh [native|all|all-major|SM]

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(CDPATH= cd -- "${script_dir}/.." && pwd)

debug_output_dir="${project_root}/cuda-backend/debug"
release_output_dir="${project_root}/cuda-backend/release"

backend_source="${project_root}/src/cudamatvec_backend.cu"
include_dir="${project_root}/src"

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    printf 'Usage: %s [native|all|all-major|SM]\n' "$0"
    exit 0
fi

if [[ $(uname -s) != "Linux" ]]; then
    printf 'ERROR: this script builds Linux/WSL CUDA .so libraries and must run on Linux.\n' >&2
    exit 1
fi

if (( $# > 1 )); then
    printf 'ERROR: expected at most one CUDA architecture argument.\n' >&2
    exit 2
fi

cuda_arch=${1:-native}
cuda_arch=${cuda_arch#sm_}

case "${cuda_arch}" in
    native|all|all-major)
        arch_flag="-arch=${cuda_arch}"
        ;;
    *[!0-9]*|'')
        printf 'ERROR: invalid CUDA architecture: %s\n' "${cuda_arch}" >&2
        exit 2
        ;;
    *)
        arch_flag="-arch=sm_${cuda_arch}"
        ;;
esac

command -v nvcc >/dev/null 2>&1 || {
    printf 'ERROR: nvcc was not found in PATH.\n' >&2
    exit 1
}

[[ -f "${backend_source}" ]] || {
    printf 'ERROR: monolithic CUDA backend source not found: %s\n' "${backend_source}" >&2
    exit 1
}

[[ -f "${include_dir}/cudamatvec_backend.h" ]] || {
    printf 'ERROR: CUDA backend header not found: %s\n' "${include_dir}/cudamatvec_backend.h" >&2
    exit 1
}

mkdir -p "${debug_output_dir}" "${release_output_dir}"

build_backend() {
    local config=$1
    local label=$2
    local define=$3
    local output=$4

    local compile_flags=()

    case "${config}" in
        debug)
            # Host debug symbols + CUDA device debug information.
            compile_flags=(-O0 -g -G)
            ;;
        release)
            compile_flags=(-O3)
            ;;
        *)
            printf 'ERROR: unknown build configuration: %s\n' "${config}" >&2
            exit 2
            ;;
    esac

    printf '[CUDA backend %s/%s] nvcc -> %s (%s)\n' \
        "${config}" "${label}" "${output}" "${arch_flag}"

    nvcc -std=c++14 --shared \
        "${compile_flags[@]}" \
        "${arch_flag}" \
        -Xcompiler=-fPIC \
        ${define} \
        -I"${include_dir}" \
        "${backend_source}" \
        -lcufft -lcublas \
        -Xlinker=-soname \
        -Xlinker="$(basename "${output}")" \
        -o "${output}"

    test -s "${output}"
}

# Debug backends.
build_backend debug double "" \
    "${debug_output_dir}/libadda_cuda_backend.so"

build_backend debug float32 "-DADDA_CUDA_SINGLE_BACKEND" \
    "${debug_output_dir}/libadda_cuda_backend_single.so"

# Release backends.
build_backend release double "" \
    "${release_output_dir}/libadda_cuda_backend.so"

build_backend release float32 "-DADDA_CUDA_SINGLE_BACKEND" \
    "${release_output_dir}/libadda_cuda_backend_single.so"

printf '\nCUDA backend libraries built successfully.\n'
printf 'Debug:\n'
printf '  %s\n' "${debug_output_dir}/libadda_cuda_backend.so"
printf '  %s\n' "${debug_output_dir}/libadda_cuda_backend_single.so"
printf 'Release:\n'
printf '  %s\n' "${release_output_dir}/libadda_cuda_backend.so"
printf '  %s\n' "${release_output_dir}/libadda_cuda_backend_single.so"
