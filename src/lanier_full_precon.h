#ifndef ADDA_LANIER_FULL_PRECON_H
#define ADDA_LANIER_FULL_PRECON_H

#include "types.h"
#include "cudamatvec_backend.h"
#include <stddef.h>

/* Build the separate DDSCAT/Lanier TQC-v1 FULL6 physical-space circulant
 * M ~= alpha_opt^-1 I - G on a direct ~1.5x auxiliary grid.  This function
 * does NOT replace LanierBuildReference().  The ADDA CUDA application wraps
 * M^-1 with S^-1 on both sides to precondition the transformed system.
 */
doublecomplex *LanierBuildFullBox(const AddaCudaLanierFullPrediction *pred,
                                  size_t phys_x,size_t phys_y,size_t phys_z,
                                  size_t *nx,size_t *ny,size_t *nz,
                                  size_t *rx,size_t *ry,size_t *rz);

doublecomplex *LanierBuildFull(const AddaCudaLanierFullPrediction *pred,
                               size_t *nx,size_t *ny,size_t *nz,
                               size_t *rx,size_t *ry,size_t *rz);

#endif
