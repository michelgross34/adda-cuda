#!/usr/bin/env bash
set -euo pipefail

# Build the two native Linux CUDA backends used by the root CMake project.
# Usage: scripts/build_cuda_backend.sh [CUDA_ARCH]
# Examples:
#   scripts/build_cuda_backend.sh native
#   scripts/build_cuda_backend.sh 70
#   scripts/build_cuda_backend.sh sm_89

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(CDPATH= cd -- "${script_dir}/.." && pwd)
output_dir="${project_root}/cuda-backend"
source_file="${project_root}/src/cudamatvec_backend.cu"
include_dir="${project_root}/src"
double_library="${output_dir}/libadda_cuda_backend.so"
single_library="${output_dir}/libadda_cuda_backend_single.so"

cuda_arch=${1:-native}
cuda_arch=${cuda_arch#sm_}
case "${cuda_arch}" in
    native|all|all-major)
        arch_flag="-arch=${cuda_arch}"
        ;;
    *[!0-9]*|'')
        printf 'ERROR: invalid CUDA architecture: %s\n' "${cuda_arch}" >&2
        printf 'Use native, all, all-major, or an SM number such as 70, 86, or 89.\n' >&2
        exit 2
        ;;
    *)
        arch_flag="-arch=sm_${cuda_arch}"
        ;;
esac

if ! command -v nvcc >/dev/null 2>&1; then
    printf 'ERROR: nvcc was not found in PATH.\n' >&2
    printf 'Install the CUDA Toolkit or add its bin directory to PATH.\n' >&2
    exit 1
fi
if [[ ! -f "${source_file}" ]]; then
    printf 'ERROR: CUDA source not found: %s\n' "${source_file}" >&2
    exit 1
fi

mkdir -p "${output_dir}"

printf '[CUDA 1/2] nvcc -> %s (%s)\n' "${double_library}" "${arch_flag}"
nvcc -std=c++14 --shared -O3 "${arch_flag}" -Xcompiler=-fPIC \
    -I"${include_dir}" "${source_file}" -lcufft -lcublas \
    -Xlinker=-soname,libadda_cuda_backend.so \
    -o "${double_library}"

printf '[CUDA 2/2] nvcc -> %s (%s)\n' "${single_library}" "${arch_flag}"
nvcc -std=c++14 --shared -O3 "${arch_flag}" -Xcompiler=-fPIC \
    -DADDA_CUDA_SINGLE_BACKEND -I"${include_dir}" "${source_file}" \
    -lcufft -lcublas -Xlinker=-soname,libadda_cuda_backend_single.so \
    -o "${single_library}"

test -s "${double_library}"
test -s "${single_library}"

printf '\nCUDA backends built successfully.\n'
printf '  double : %s\n' "${double_library}"
printf '  float32: %s\n' "${single_library}"
printf '\nReload CMake in CLion, then use Build > Build All in Debug.\n'
