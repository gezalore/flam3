/*
    FLAM3 - cosmic recursive fractal flames
    Copyright (C) 1992-2009 Spotworks LLC

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef variations_included
#define variations_included

#include "immintrin.h"

#include "private.h"

typedef __attribute__((aligned(32)))  struct {

  __m128d t; /* Starting coordinates */
  __m128d p; /* Output coordinates */

  double precalc_atan, precalc_sina; /* Precalculated, if needed */
  double precalc_cosa, precalc_sqrt;
  double precalc_sumsq, precalc_atanyx;

  flam3_xform *xform; /* For the important values */



  /* Pointer to the isaac RNG state */
  randctx *rc;

} flam3_iter_helper;

void xform_precalc(flam3_genome *cp, int xi);
int prepare_precalc_flags(flam3_genome *);

__m256d apply_xform(flam3_genome *cp, int fn, const __m256d p, randctx *rc,
    int *bad, int consec);
void initialize_xforms(flam3_genome *thiscp, int start_here);
#endif
