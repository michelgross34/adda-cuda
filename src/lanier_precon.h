#ifndef ADDA_LANIER_PRECON_H
#define ADDA_LANIER_PRECON_H

#include "types.h"
#include <stddef.h>

/* Build the homogeneous three-level Chan/Lanier circulant approximation of
 * ADDA's transformed operator A = I + S D S.
 *
 * expansion=1.0: direct reference grid at least as large as the physical box
 *                (odd dimensions are rounded upward to even lengths).
 * expansion=1.5: direct auxiliary grid ceil(1.5*box) rounded upward to even
 *                lengths, matching the validated DDSCAT Lanier direct path.
 *
 * Output layout is six reduced Fourier-octant components, component-major:
 *   [xx][xy][xz][yy][yz][zz]
 * with each component indexed as ((kx*ry)+ky)*rz+kz.
 * The returned inverse blocks already include the 1/(nx*ny*nz) factor needed
 * by an unnormalised inverse cuFFT. Caller owns the malloc()'d result.
 */
doublecomplex *LanierBuildReference(double expansion,
                                     size_t *nx,size_t *ny,size_t *nz,
                                     size_t *rx,size_t *ry,size_t *rz);

#endif
