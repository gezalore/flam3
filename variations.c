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

#include "variations.h"
#include "interpolation.h" 

#define badvalue(x) (((x)!=(x))||((x)>1e10)||((x)<-1e10))

/* Wrap the sincos function for Macs */
#if defined(__APPLE__) || defined(_MSC_VER)
#define sincos(x,s,c) *(s)=sin(x); *(c)=cos(x);
#else
extern void sincos(double x, double *s, double *c);
#endif

#ifdef _MSC_VER
#define trunc (int)
#endif

char *flam3_variation_names[1+flam3_nvariations] = {
  "linear",
  "sinusoidal",
  "spherical",
  "swirl",
  "horseshoe",
  "polar",
  "handkerchief",
  "heart",
  "disc",
  "spiral",
  "hyperbolic",
  "diamond",
  "ex",
  "julia",
  "bent",
  "waves",
  "fisheye",
  "popcorn",
  "exponential",
  "power",
  "cosine",
  "rings",
  "fan",
  "blob",
  "pdj",
  "fan2",
  "rings2",
  "eyefish",
  "bubble",
  "cylinder",
  "perspective",
  "noise",
  "julian",
  "juliascope",
  "blur",
  "gaussian_blur",
  "radial_blur",
  "pie",
  "ngon",
  "curl",
  "rectangles",
  "arch",
  "tangent",
  "square",
  "rays",
  "blade",
  "secant2",
  "twintrian",
  "cross",
  "disc2",
  "super_shape",
  "flower",
  "conic",
  "parabola",
  "bent2",
  "bipolar",
  "boarders",
  "butterfly",
  "cell",
  "cpow",
  "curve",
  "edisc",
  "elliptic",
  "escher",
  "foci",
  "lazysusan",
  "loonie",
  "pre_blur",
  "modulus",
  "oscilloscope",
  "polar2",
  "popcorn2",
  "scry",
  "separation",
  "split",
  "splits",
  "stripes",
  "wedge",
  "wedge_julia",
  "wedge_sph",
  "whorl",
  "waves2",
  "exp",
  "log",
  "sin",
  "cos",
  "tan",
  "sec",
  "csc",
  "cot",
  "sinh",
  "cosh",
  "tanh",
  "sech",
  "csch",
  "coth",
  "auger",
  "flux",
  "mobius",
  0
};

/*
 * VARIATION FUNCTIONS
 * must be of the form void (void *, double)
 */
static void var0_linear(flam3_iter_helper *f, double weight) {
   /* linear */
   /* nx = tx;
      ny = ty;
      p[0] += v * nx;
      p[1] += v * ny; */

  f->p[0] += weight * f->t[0];
  f->p[1] += weight * f->t[1];
}

static void var1_sinusoidal(flam3_iter_helper *f, double weight) {
   /* sinusoidal */
   /* nx = sin(tx);
      ny = sin(ty);
      p[0] += v * nx;
      p[1] += v * ny; */

  f->p[0] += weight * sin(f->t[0]);
  f->p[1] += weight * sin(f->t[1]);
}

static void var2_spherical(flam3_iter_helper *f, double weight) {
   /* spherical */
   /* double r2 = tx * tx + ty * ty + 1e-6;
      nx = tx / r2;
      ny = ty / r2;
      p[0] += v * nx;
      p[1] += v * ny; */

//  double r2 = weight / (f->precalc_v_sumsq[0] + EPS);
//
//  f->p[0] += r2 * f->t[0];
//  f->p[1] += r2 * f->t[1];

  const __m128d v_w = _mm_set1_pd(weight);
  const __m128d v_EPS = _mm_set1_pd(EPS);
  const __m128d v_den = _mm_add_pd(f->precalc_v_sumsq, v_EPS);
  const __m128d v_r2 = _mm_div_pd(v_w, v_den);
  const __m128d v_dp = _mm_mul_pd(v_r2, f->t);
  f->p = _mm_add_pd(f->p, v_dp);
}

static void var3_swirl(flam3_iter_helper *f, double weight) {
   /* swirl */
   /* double r2 = tx * tx + ty * ty;    /k here is fun
      double c1 = sin(r2);
      double c2 = cos(r2);
      nx = c1 * tx - c2 * ty;
      ny = c2 * tx + c1 * ty;
      p[0] += v * nx;
      p[1] += v * ny; */

  double r2 = f->precalc_v_sumsq[0];
   double c1,c2;
   double nx,ny;
   
   sincos(r2,&c1,&c2);
//   double c1 = sin(r2);
//   double c2 = cos(r2);
  nx = c1 * f->t[0] - c2 * f->t[1];
  ny = c2 * f->t[0] + c1 * f->t[1];

  f->p[0] += weight * nx;
  f->p[1] += weight * ny;
}

static void var4_horseshoe(flam3_iter_helper *f, double weight) {
   /* horseshoe */
   /* a = atan2(tx, ty);
      c1 = sin(a);
      c2 = cos(a);
      nx = c1 * tx - c2 * ty;
      ny = c2 * tx + c1 * ty;
      p[0] += v * nx;
      p[1] += v * ny;  */

  double r = weight / (f->precalc_v_sqrt[0] + EPS);

  f->p[0] += (f->t[0] - f->t[1]) * (f->t[0] + f->t[1]) * r;
  f->p[1] += 2.0 * f->t[0] * f->t[1] * r;
}

static void var5_polar(flam3_iter_helper *f, double weight) {
   /* polar */
   /* nx = atan2(tx, ty) / M_PI;
      ny = sqrt(tx * tx + ty * ty) - 1.0;
      p[0] += v * nx;
      p[1] += v * ny; */

   double nx = f->precalc_atan * M_1_PI;
  double ny = f->precalc_v_sqrt[0] - 1.0;

  f->p[0] += weight * nx;
  f->p[1] += weight * ny;
}

static void var6_handkerchief(flam3_iter_helper *f, double weight) {
   /* folded handkerchief */
   /* a = atan2(tx, ty);
      r = sqrt(tx*tx + ty*ty);
      p[0] += v * sin(a+r) * r;
      p[1] += v * cos(a-r) * r; */

   double a = f->precalc_atan;
  double r = f->precalc_v_sqrt[0];

  f->p[0] += weight * r * sin(a + r);
  f->p[1] += weight * r * cos(a - r);
}

static void var7_heart(flam3_iter_helper *f, double weight) {
   /* heart */
   /* a = atan2(tx, ty);
      r = sqrt(tx*tx + ty*ty);
      a *= r;
      p[0] += v * sin(a) * r;
      p[1] += v * cos(a) * -r; */

  double a = f->precalc_v_sqrt[0] * f->precalc_atan;
   double ca,sa;
  double r = weight * f->precalc_v_sqrt[0];

   sincos(a,&sa,&ca);

  f->p[0] += r * sa;
  f->p[1] += (-r) * ca;
}

static void var8_disc(flam3_iter_helper *f, double weight) {
   /* disc */
   /* nx = tx * M_PI;
      ny = ty * M_PI;
      a = atan2(nx, ny);
      r = sqrt(nx*nx + ny*ny);
      p[0] += v * sin(r) * a / M_PI;
      p[1] += v * cos(r) * a / M_PI; */

   double a = f->precalc_atan * M_1_PI;
  double r = M_PI * f->precalc_v_sqrt[0];
   double sr,cr;
   sincos(r,&sr,&cr);

  f->p[0] += weight * sr * a;
  f->p[1] += weight * cr * a;
}

static void var9_spiral(flam3_iter_helper *f, double weight) {
   /* spiral */
   /* a = atan2(tx, ty);
      r = sqrt(tx*tx + ty*ty) + 1e-6;
      p[0] += v * (cos(a) + sin(r)) / r;
      p[1] += v * (sin(a) - cos(r)) / r; */

  double r = f->precalc_v_sqrt[0] + EPS;
   double r1 = weight/r;
   double sr,cr;
   sincos(r,&sr,&cr);

  f->p[0] += r1 * (f->precalc_cosa + sr);
  f->p[1] += r1 * (f->precalc_sina - cr);
}

static void var10_hyperbolic(flam3_iter_helper *f, double weight) {
   /* hyperbolic */
   /* a = atan2(tx, ty);
      r = sqrt(tx*tx + ty*ty) + 1e-6;
      p[0] += v * sin(a) / r;
      p[1] += v * cos(a) * r; */

  double r = f->precalc_v_sqrt[0] + EPS;

  f->p[0] += weight * f->precalc_sina / r;
  f->p[1] += weight * f->precalc_cosa * r;
}

static void var11_diamond(flam3_iter_helper *f, double weight) {
   /* diamond */
   /* a = atan2(tx, ty);
      r = sqrt(tx*tx + ty*ty);
      p[0] += v * sin(a) * cos(r);
      p[1] += v * cos(a) * sin(r); */

  double r = f->precalc_v_sqrt[0];
   double sr,cr;
   sincos(r,&sr,&cr);

  f->p[0] += weight * f->precalc_sina * cr;
  f->p[1] += weight * f->precalc_cosa * sr;
}

static void var12_ex(flam3_iter_helper *f, double weight) {
   /* ex */
   /* a = atan2(tx, ty);
      r = sqrt(tx*tx + ty*ty);
      n0 = sin(a+r);
      n1 = cos(a-r);
      m0 = n0 * n0 * n0 * r;
      m1 = n1 * n1 * n1 * r;
      p[0] += v * (m0 + m1);
      p[1] += v * (m0 - m1); */

   double a = f->precalc_atan;
  double r = f->precalc_v_sqrt[0];

   double n0 = sin(a+r);
   double n1 = cos(a-r);

   double m0 = n0 * n0 * n0 * r;
   double m1 = n1 * n1 * n1 * r;

  f->p[0] += weight * (m0 + m1);
  f->p[1] += weight * (m0 - m1);
}

static void var13_julia(flam3_iter_helper *f, double weight) {
   /* julia */
   /* a = atan2(tx, ty)/2.0;
      if (flam3_random_bit()) a += M_PI;
      r = pow(tx*tx + ty*ty, 0.25);
      nx = r * cos(a);
      ny = r * sin(a);
      p[0] += v * nx;
      p[1] += v * ny; */

   double r;
   double a = 0.5 * f->precalc_atan;
   double sa,ca;

   if (flam3_random_isaac_bit(f->rc)) //(flam3_random_bit())
      a += M_PI;

  r = weight * sqrt(f->precalc_v_sqrt[0]);

   sincos(a,&sa,&ca);

  f->p[0] += r * ca;
  f->p[1] += r * sa;
}

static void var14_bent(flam3_iter_helper *f, double weight) {
   /* bent */
   /* nx = tx;
      ny = ty;
      if (nx < 0.0) nx = nx * 2.0;
      if (ny < 0.0) ny = ny / 2.0;
      p[0] += v * nx;
      p[1] += v * ny; */

  double nx = f->t[0];
  double ny = f->t[1];

   if (nx < 0.0)
      nx = nx * 2.0;
   if (ny < 0.0)
      ny = ny / 2.0;

  f->p[0] += weight * nx;
  f->p[1] += weight * ny;
}

static void var15_waves(flam3_iter_helper *f, double weight) {
   /* waves */
   /* dx = coef[2][0];
      dy = coef[2][1];
      nx = tx + coef[1][0]*sin(ty/((dx*dx)+EPS));
      ny = ty + coef[1][1]*sin(tx/((dy*dy)+EPS));
      p[0] += v * nx;
      p[1] += v * ny; */

   double c10 = f->xform->c[1][0];
   double c11 = f->xform->c[1][1];

  double nx = f->t[0] + c10 * sin(f->t[1] * f->xform->waves_dx2);
  double ny = f->t[1] + c11 * sin(f->t[0] * f->xform->waves_dy2);

  f->p[0] += weight * nx;
  f->p[1] += weight * ny;
}

static void var16_fisheye(flam3_iter_helper *f, double weight) {
   /* fisheye */
   /* a = atan2(tx, ty);
      r = sqrt(tx*tx + ty*ty);
      r = 2 * r / (r + 1);
      nx = r * cos(a);
      ny = r * sin(a);
      p[0] += v * nx;
      p[1] += v * ny; */

  double r = f->precalc_v_sqrt[0];

   r = 2 * weight / (r+1);

  f->p[0] += r * f->t[1];
  f->p[1] += r * f->t[0];
}

static void var17_popcorn(flam3_iter_helper *f, double weight) {
   /* popcorn */
   /* dx = tan(3*ty);
      dy = tan(3*tx);
      nx = tx + coef[2][0] * sin(dx);
      ny = ty + coef[2][1] * sin(dy);
      p[0] += v * nx;
      p[1] += v * ny; */

  double dx = tan(3 * f->t[1]);
  double dy = tan(3 * f->t[0]);

  double nx = f->t[0] + f->xform->c[2][0] * sin(dx);
  double ny = f->t[1] + f->xform->c[2][1] * sin(dy);

  f->p[0] += weight * nx;
  f->p[1] += weight * ny;
}

static void var18_exponential(flam3_iter_helper *f, double weight) {
   /* exponential */
   /* dx = exp(tx-1.0);
      dy = M_PI * ty;
      nx = cos(dy) * dx;
      ny = sin(dy) * dx;
      p[0] += v * nx;
      p[1] += v * ny; */

  double dx = weight * exp(f->t[0] - 1.0);
  double dy = M_PI * f->t[1];
   double sdy,cdy;

   sincos(dy,&sdy,&cdy);
   

   f->p[0] += dx * cdy;
  f->p[1] += dx * sdy;
}

static void var19_power(flam3_iter_helper *f, double weight) {
   /* power */
   /* a = atan2(tx, ty);
      sa = sin(a);
      r = sqrt(tx*tx + ty*ty);
      r = pow(r, sa);
      nx = r * precalc_cosa;
      ny = r * sa;
      p[0] += v * nx;
      p[1] += v * ny; */

  double r = weight * pow(f->precalc_v_sqrt[0], f->precalc_sina);

  f->p[0] += r * f->precalc_cosa;
  f->p[1] += r * f->precalc_sina;
}

static void var20_cosine(flam3_iter_helper *f, double weight) {
   /* cosine */
   /* nx = cos(tx * M_PI) * cosh(ty);
      ny = -sin(tx * M_PI) * sinh(ty);
      p[0] += v * nx;
      p[1] += v * ny; */

  double a = f->t[0] * M_PI;
   double sa,ca;
   double nx,ny;
   
   sincos(a,&sa,&ca);
  nx = ca * cosh(f->t[1]);
  ny = -sa * sinh(f->t[1]);

  f->p[0] += weight * nx;
  f->p[1] += weight * ny;
}

static void var21_rings(flam3_iter_helper *f, double weight) {
   /* rings */
   /* dx = coef[2][0];
      dx = dx * dx + EPS;
      r = sqrt(tx*tx + ty*ty);
      r = fmod(r + dx, 2*dx) - dx + r*(1-dx);
      a = atan2(tx, ty);
      nx = cos(a) * r;
      ny = sin(a) * r;
      p[0] += v * nx;
      p[1] += v * ny; */

   double dx = f->xform->c[2][0] * f->xform->c[2][0] + EPS;
  double r = f->precalc_v_sqrt[0];
   r = weight * (fmod(r+dx, 2*dx) - dx + r * (1 - dx));

  f->p[0] += r * f->precalc_cosa;
  f->p[1] += r * f->precalc_sina;
}

static void var22_fan(flam3_iter_helper *f, double weight) {
   /* fan */
   /* dx = coef[2][0];
      dy = coef[2][1];
      dx = M_PI * (dx * dx + EPS);
      dx2 = dx/2;
      a = atan(tx,ty);
      r = sqrt(tx*tx + ty*ty);
      a += (fmod(a+dy, dx) > dx2) ? -dx2 : dx2;
      nx = cos(a) * r;
      ny = sin(a) * r;
      p[0] += v * nx;
      p[1] += v * ny; */

   double dx = M_PI * (f->xform->c[2][0] * f->xform->c[2][0] + EPS);
   double dy = f->xform->c[2][1];
   double dx2 = 0.5 * dx;

   double a = f->precalc_atan;
  double r = weight * f->precalc_v_sqrt[0];
   double sa,ca;

   a += (fmod(a+dy,dx) > dx2) ? -dx2 : dx2;
   sincos(a,&sa,&ca);

  f->p[0] += r * ca;
  f->p[1] += r * sa;
}

static void var23_blob(flam3_iter_helper *f, double weight) {
   /* blob */
   /* a = atan2(tx, ty);
      r = sqrt(tx*tx + ty*ty);
      r = r * (bloblow + (blobhigh-bloblow) * (0.5 + 0.5 * sin(blobwaves * a)));
      nx = sin(a) * r;
      ny = cos(a) * r;

      p[0] += v * nx;
      p[1] += v * ny; */

  double r = f->precalc_v_sqrt[0];
   double a = f->precalc_atan;
   double bdiff = f->xform->blob_high - f->xform->blob_low;

   r = r * (f->xform->blob_low +
            bdiff * (0.5 + 0.5 * sin(f->xform->blob_waves * a)));

  f->p[0] += weight * f->precalc_sina * r;
  f->p[1] += weight * f->precalc_cosa * r;
}

static void var24_pdj(flam3_iter_helper *f, double weight) {
   /* pdj */
   /* nx1 = cos(pdjb * tx);
      nx2 = sin(pdjc * tx);
      ny1 = sin(pdja * ty);
      ny2 = cos(pdjd * ty);

      p[0] += v * (ny1 - nx1);
      p[1] += v * (nx2 - ny2); */

  double nx1 = cos(f->xform->pdj_b * f->t[0]);
  double nx2 = sin(f->xform->pdj_c * f->t[0]);
  double ny1 = sin(f->xform->pdj_a * f->t[1]);
  double ny2 = cos(f->xform->pdj_d * f->t[1]);

  f->p[0] += weight * (ny1 - nx1);
  f->p[1] += weight * (nx2 - ny2);
}

static void var25_fan2(flam3_iter_helper *f, double weight) {
   /* fan2 */
   /* a = precalc_atan;
   r = precalc_v_sqrt[0];

      dy = fan2y;
      dx = M_PI * (fan2x * fan2x + EPS);
      dx2 = dx / 2.0;

      t = a + dy - dx * (int)((a + dy)/dx);

      if (t > dx2)
         a = a - dx2;
      else
         a = a + dx2;

      nx = sin(a) * r;
      ny = cos(a) * r;

      p[0] += v * nx;
      p[1] += v * ny; */

   double dy = f->xform->fan2_y;
   double dx = M_PI * (f->xform->fan2_x * f->xform->fan2_x + EPS);
   double dx2 = 0.5 * dx;
   double a = f->precalc_atan;
   double sa,ca;
  double r = weight * f->precalc_v_sqrt[0];

   double t = a + dy - dx * (int)((a + dy)/dx);

   if (t>dx2)
      a = a-dx2;
   else
      a = a+dx2;
      
   sincos(a,&sa,&ca);

  f->p[0] += r * sa;
  f->p[1] += r * ca;
}

static void var26_rings2(flam3_iter_helper *f, double weight) {
   /* rings2 */
  /* r = precalc_v_sqrt[0];
      dx = rings2val * rings2val + EPS;
      r += dx - 2.0*dx*(int)((r + dx)/(2.0 * dx)) - dx + r * (1.0-dx);
      nx = precalc_sina * r;
      ny = precalc_cosa * r;
      p[0] += v * nx;
      p[1] += v * ny; */

  double r = f->precalc_v_sqrt[0];
   double dx = f->xform->rings2_val * f->xform->rings2_val + EPS;

   r += -2.0*dx*(int)((r+dx)/(2.0*dx)) + r * (1.0-dx);

  f->p[0] += weight * f->precalc_sina * r;
  f->p[1] += weight * f->precalc_cosa * r;
}

static void var27_eyefish(flam3_iter_helper *f, double weight) {
   /* eyefish */
  /* r = 2.0 * v / (precalc_v_sqrt[0] + 1.0);
      p[0] += r*tx;
      p[1] += r*ty; */

  double r = (weight * 2.0) / (f->precalc_v_sqrt[0] + 1.0);

  f->p[0] += r * f->t[0];
  f->p[1] += r * f->t[1];
}

static void var28_bubble(flam3_iter_helper *f, double weight) {
   /* bubble */

  double r = weight / (0.25 * (f->precalc_v_sumsq[0]) + 1);

  f->p[0] += r * f->t[0];
  f->p[1] += r * f->t[1];
}

static void var29_cylinder(flam3_iter_helper *f, double weight) {
   /* cylinder (01/06) */

  f->p[0] += weight * sin(f->t[0]);
  f->p[1] += weight * f->t[1];
}

static void var30_perspective(flam3_iter_helper *f, double weight) {
   /* perspective (01/06) */

  double t = 1.0
      / (f->xform->perspective_dist - f->t[1] * f->xform->persp_vsin);

  f->p[0] += weight * f->xform->perspective_dist * f->t[0] * t;
  f->p[1] += weight * f->xform->persp_vfcos * f->t[1] * t;
}

static void var31_noise(flam3_iter_helper *f, double weight) {
   /* noise (03/06) */

   double tmpr, sinr, cosr, r;

   tmpr = flam3_random_isaac_01(f->rc) * 2 * M_PI;
   sincos(tmpr,&sinr,&cosr);

   r = weight * flam3_random_isaac_01(f->rc);

  f->p[0] += f->t[0] * r * cosr;
  f->p[1] += f->t[1] * r * sinr;
}

static void var32_juliaN_generic(flam3_iter_helper *f, double weight) {
   /* juliaN (03/06) */

   int t_rnd = trunc((f->xform->julian_rN)*flam3_random_isaac_01(f->rc));
   
   double tmpr = (f->precalc_atanyx + 2 * M_PI * t_rnd) / f->xform->julian_power;

  double r = weight * pow(f->precalc_v_sumsq[0], f->xform->julian_cn);
   double sina, cosa;
   sincos(tmpr,&sina,&cosa);

  f->p[0] += r * cosa;
  f->p[1] += r * sina;
}

static void var33_juliaScope_generic(flam3_iter_helper *f, double weight) {
   /* juliaScope (03/06) */

   int t_rnd = trunc((f->xform->juliascope_rN) * flam3_random_isaac_01(f->rc));

   double tmpr, r;
   double sina, cosa;

   if ((t_rnd & 1) == 0)
      tmpr = (2 * M_PI * t_rnd + f->precalc_atanyx) / f->xform->juliascope_power;
   else
      tmpr = (2 * M_PI * t_rnd - f->precalc_atanyx) / f->xform->juliascope_power;

   sincos(tmpr,&sina,&cosa);

  r = weight * pow(f->precalc_v_sumsq[0], f->xform->juliascope_cn);

  f->p[0] += r * cosa;
  f->p[1] += r * sina;
}

static void var34_blur(flam3_iter_helper *f, double weight) {
   /* blur (03/06) */

   double tmpr, sinr, cosr, r;

   tmpr = flam3_random_isaac_01(f->rc) * 2 * M_PI;
   sincos(tmpr,&sinr,&cosr);

   r = weight * flam3_random_isaac_01(f->rc);

  f->p[0] += r * cosr;
  f->p[1] += r * sinr;
}

static void var35_gaussian(flam3_iter_helper *f, double weight) {
   /* gaussian (09/06) */

   double ang, r, sina, cosa;

   ang = flam3_random_isaac_01(f->rc) * 2 * M_PI;
   sincos(ang,&sina,&cosa);

   r = weight * ( flam3_random_isaac_01(f->rc) + flam3_random_isaac_01(f->rc)
                   + flam3_random_isaac_01(f->rc) + flam3_random_isaac_01(f->rc) - 2.0 );

  f->p[0] += r * cosa;
  f->p[1] += r * sina;
}

static void var36_radial_blur(flam3_iter_helper *f, double weight) {
   /* radial blur (09/06) */
   /* removed random storage 6/07 */

   double rndG, ra, rz, tmpa, sa, ca;

   /* Get pseudo-gaussian */
   rndG = weight * (flam3_random_isaac_01(f->rc) + flam3_random_isaac_01(f->rc)
                   + flam3_random_isaac_01(f->rc) + flam3_random_isaac_01(f->rc) - 2.0);

   /* Calculate angle & zoom */
  ra = f->precalc_v_sqrt[0];
   tmpa = f->precalc_atanyx + f->xform->radialBlur_spinvar*rndG;
   sincos(tmpa,&sa,&ca);
   rz = f->xform->radialBlur_zoomvar * rndG - 1;

  f->p[0] += ra * ca + rz * f->t[0];
  f->p[1] += ra * sa + rz * f->t[1];
}

static void var37_pie(flam3_iter_helper *f, double weight) {
   /* pie by Joel Faber (June 2006) */

   double a, r, sa, ca;
   int sl;

   sl = (int) (flam3_random_isaac_01(f->rc) * f->xform->pie_slices + 0.5);
   a = f->xform->pie_rotation +
       2.0 * M_PI * (sl + flam3_random_isaac_01(f->rc) * f->xform->pie_thickness) / f->xform->pie_slices;
   r = weight * flam3_random_isaac_01(f->rc);
   sincos(a,&sa,&ca);

  f->p[0] += r * ca;
  f->p[1] += r * sa;
}

static void var38_ngon(flam3_iter_helper *f, double weight) {
   /* ngon by Joel Faber (09/06) */

   double r_factor,theta,phi,b, amp;

  r_factor = pow(f->precalc_v_sumsq[0], f->xform->ngon_power / 2.0);

   theta = f->precalc_atanyx;
   b = 2*M_PI/f->xform->ngon_sides;

   phi = theta - (b*floor(theta/b));
   if (phi > b/2)
      phi -= b;

   amp = f->xform->ngon_corners * (1.0 / (cos(phi) + EPS) - 1.0) + f->xform->ngon_circle;
   amp /= (r_factor + EPS);

  f->p[0] += weight * f->t[0] * amp;
  f->p[1] += weight * f->t[1] * amp;
}

static void var39_curl(flam3_iter_helper *f, double weight)
{
  double re = 1.0 + f->xform->curl_c1 * f->t[0]
      + f->xform->curl_c2 * (f->t[0] * f->t[0] - f->t[1] * f->t[1]);
  double im = f->xform->curl_c1 * f->t[1]
      + 2.0 * f->xform->curl_c2 * f->t[0] * f->t[1];

    double r = weight / (re*re + im*im);

  f->p[0] += (f->t[0] * re + f->t[1] * im) * r;
  f->p[1] += (f->t[1] * re - f->t[0] * im) * r;
}

static void var40_rectangles(flam3_iter_helper *f, double weight)
{
    if (f->xform->rectangles_x==0)
    f->p[0] += weight * f->t[0];
    else
    f->p[0] += weight
        * ((2 * floor(f->t[0] / f->xform->rectangles_x) + 1)
            * f->xform->rectangles_x - f->t[0]);

    if (f->xform->rectangles_y==0)
    f->p[1] += weight * f->t[1];
    else
    f->p[1] += weight
        * ((2 * floor(f->t[1] / f->xform->rectangles_y) + 1)
            * f->xform->rectangles_y - f->t[1]);

}

static void var41_arch(flam3_iter_helper *f, double weight)
{
   /* Z+ variation Jan 07
   procedure TXForm.Arch;
   var
     sinr, cosr: double;
   begin
     SinCos(random * vars[29]*pi, sinr, cosr);
     FPx := FPx + sinr*vars[29];
     FPy := FPy + sqr(sinr)/cosr*vars[29];
   end;
   */
   
   /*
    * !!! Note !!!
    * This code uses the variation weight in a non-standard fashion, and
    * it may change or even be removed in future versions of flam3.
    */

   double ang = flam3_random_isaac_01(f->rc) * weight * M_PI;
   double sinr,cosr;
   sincos(ang,&sinr,&cosr);

  f->p[0] += weight * sinr;
  f->p[1] += weight * (sinr * sinr) / cosr;

}

static void var42_tangent(flam3_iter_helper *f, double weight)
{
   /* Z+ variation Jan 07
   procedure TXForm.Tangent;
   begin
     FPx := FPx + vars[30] * (sin(FTx)/cos(FTy));
     FPy := FPy + vars[30] * (sin(FTy)/cos(FTy));
   end;
   */

  f->p[0] += weight * sin(f->t[0]) / cos(f->t[1]);
  f->p[1] += weight * tan(f->t[1]);

}

static void var43_square(flam3_iter_helper *f, double weight)
{
   /* Z+ variation Jan 07
   procedure TXForm.SquareBlur;
   begin
     FPx := FPx + vars[31] * (random - 0.5);
     FPy := FPy + vars[31] * (random - 0.5);
   end;
   */

  f->p[0] += weight * (flam3_random_isaac_01(f->rc) - 0.5);
  f->p[1] += weight * (flam3_random_isaac_01(f->rc) - 0.5);

}

static void var44_rays(flam3_iter_helper *f, double weight)
{
   /* Z+ variation Jan 07
   procedure TXForm.Rays;
   var
     r, sinr, cosr, tgr: double;
   begin
     SinCos(random * vars[32]*pi, sinr, cosr);
     r := vars[32] / (sqr(FTx) + sqr(FTy) + EPS);
     tgr := sinr/cosr;
     FPx := FPx + tgr * (cos(FTx)*vars[32]) * r;
     FPy := FPy + tgr * (sin(FTy)*vars[32]) * r;
   end;
   */

   /*
    * !!! Note !!!
    * This code uses the variation weight in a non-standard fashion, and
    * it may change or even be removed in future versions of flam3.
    */

   double ang = weight * flam3_random_isaac_01(f->rc) * M_PI;
  double r = weight / (f->precalc_v_sumsq[0] + EPS);
   double tanr = weight * tan(ang) * r;


   f->p[0] += tanr * cos(f->t[0]);
  f->p[1] += tanr * sin(f->t[1]);

}

static void var45_blade(flam3_iter_helper *f, double weight)
{
   /* Z+ variation Jan 07
   procedure TXForm.Blade;
   var
     r, sinr, cosr: double;
   begin
     r := sqrt(sqr(FTx) + sqr(FTy))*vars[33];
     SinCos(r*random, sinr, cosr);
     FPx := FPx + vars[33] * FTx * (cosr + sinr);
     FPy := FPy + vars[33] * FTx * (cosr - sinr);
   end;
   */

   /*
    * !!! Note !!!
    * This code uses the variation weight in a non-standard fashion, and
    * it may change or even be removed in future versions of flam3.
    */

  double r = flam3_random_isaac_01(f->rc) * weight * f->precalc_v_sqrt[0];
   double sinr,cosr;

   sincos(r,&sinr,&cosr);

  f->p[0] += weight * f->t[0] * (cosr + sinr);
  f->p[1] += weight * f->t[0] * (cosr - sinr);

}

static void var46_secant2(flam3_iter_helper *f, double weight)
{
   /* Intended as a 'fixed' version of secant */

   /*
    * !!! Note !!!
    * This code uses the variation weight in a non-standard fashion, and
    * it may change or even be removed in future versions of flam3.
    */

  double r = weight * f->precalc_v_sqrt[0];
   double cr = cos(r);
   double icr = 1.0/cr;

  f->p[0] += weight * f->t[0];

   if (cr<0)
    f->p[1] += weight * (icr + 1);
   else
    f->p[1] += weight * (icr - 1);
}

static void var47_twintrian(flam3_iter_helper *f, double weight)
{
   /* Z+ variation Jan 07
   procedure TXForm.TwinTrian;
   var
     r, diff, sinr, cosr: double;
   begin
     r := sqrt(sqr(FTx) + sqr(FTy))*vars[35];
     SinCos(r*random, sinr, cosr);
     diff := Math.Log10(sinr*sinr)+cosr;
     FPx := FPx + vars[35] * FTx * diff;
     FPy := FPy + vars[35] * FTx * (diff - (sinr*pi));
   end;
   */

   /*
    * !!! Note !!!
    * This code uses the variation weight in a non-standard fashion, and
    * it may change or even be removed in future versions of flam3.
    */

  double r = flam3_random_isaac_01(f->rc) * weight * f->precalc_v_sqrt[0];
   double sinr,cosr,diff;

   sincos(r,&sinr,&cosr);
   diff = log10(sinr*sinr)+cosr;
   
   if (badvalue(diff))
      diff = -30.0;      

   f->p[0] += weight * f->t[0] * diff;
  f->p[1] += weight * f->t[0] * (diff - sinr * M_PI);

}

static void var48_cross(flam3_iter_helper *f, double weight)
{
   /* Z+ variation Jan 07
   procedure TXForm.Cross;
   var
     r: double;
   begin
     r := vars[36]*sqrt(1/(sqr(sqr(FTx)-sqr(FTy))+EPS));
     FPx := FPx + FTx * r;
     FPy := FPy + FTy * r;
   end;
   */

  double s = f->t[0] * f->t[0] - f->t[1] * f->t[1];
   double r = weight * sqrt(1.0 / (s*s+EPS));

  f->p[0] += f->t[0] * r;
  f->p[1] += f->t[1] * r;

}

static void var49_disc2(flam3_iter_helper *f, double weight)
{
   /* Z+ variation Jan 07
   c := vvar/PI;
   k := rot*PI;
     sinadd := Sin(add);
     cosadd := Cos(add);
   cosadd := cosadd - 1;
   if (add > 2*PI) then begin
     cosadd := cosadd * (1 + add - 2*PI);
     sinadd := sinadd * (1 + add - 2*PI)
   end
   else if (add < -2*PI) then begin
     cosadd := cosadd * (1 + add + 2*PI);
     sinadd := sinadd * (1 + add + 2*PI)
   end
   end;
   procedure TVariationDisc2.CalcFunction;
   var
     r, sinr, cosr: extended;
   begin
     SinCos(k * (FTx^+FTy^), sinr, cosr);   //rot*PI
     r := c * arctan2(FTx^, FTy^); //vvar/PI
     FPx^ := FPx^ + (sinr + cosadd) * r;
     FPy^ := FPy^ + (cosr + sinadd) * r;
   */

   double r,t,sinr, cosr;

  t = f->xform->disc2_timespi * (f->t[0] + f->t[1]);
   sincos(t,&sinr,&cosr);
   r = weight * f->precalc_atan / M_PI;

  f->p[0] += (sinr + f->xform->disc2_cosadd) * r;
  f->p[1] += (cosr + f->xform->disc2_sinadd) * r;

}

static void var50_supershape(flam3_iter_helper *f, double weight) {

   double theta;
   double t1,t2,r;
   double st,ct;
   double myrnd;

   theta = f->xform->super_shape_pm_4 * f->precalc_atanyx + M_PI_4;
   
   sincos(theta,&st,&ct);

   t1 = fabs(ct);
   t1 = pow(t1,f->xform->super_shape_n2);

   t2 = fabs(st);
   t2 = pow(t2,f->xform->super_shape_n3);
   
   myrnd = f->xform->super_shape_rnd;

  r = weight
      * ((myrnd * flam3_random_isaac_01(f->rc)
          + (1.0 - myrnd) * f->precalc_v_sqrt[0]) - f->xform->super_shape_holes)
      * pow(t1 + t2, f->xform->super_shape_pneg1_n1) / f->precalc_v_sqrt[0];

  f->p[0] += r * f->t[0];
  f->p[1] += r * f->t[1];
}

static void var51_flower(flam3_iter_helper *f, double weight) {
    /* cyberxaos, 4/2007 */
    /*   theta := arctan2(FTy^, FTx^);
         r := (random-holes)*cos(petals*theta);
         FPx^ := FPx^ + vvar*r*cos(theta);
         FPy^ := FPy^ + vvar*r*sin(theta);*/
 
    double theta = f->precalc_atanyx;
    double r = weight * (flam3_random_isaac_01(f->rc) - f->xform->flower_holes) * 
                    cos(f->xform->flower_petals * theta) / f->precalc_v_sqrt[0];

  f->p[0] += r * f->t[0];
  f->p[1] += r * f->t[1];
}

static void var52_conic(flam3_iter_helper *f, double weight) {
    /* cyberxaos, 4/2007 */
    /*   theta := arctan2(FTy^, FTx^);
         r :=  (random - holes)*((eccentricity)/(1+eccentricity*cos(theta)));
         FPx^ := FPx^ + vvar*r*cos(theta);
         FPy^ := FPy^ + vvar*r*sin(theta); */
 
    double ct = f->t[0] / f->precalc_v_sqrt[0];
  double r = weight * (flam3_random_isaac_01(f->rc) - f->xform->conic_holes)
      *
                    f->xform->conic_eccentricity / (1 + f->xform->conic_eccentricity * ct)
      / f->precalc_v_sqrt[0];

  f->p[0] += r * f->t[0];
  f->p[1] += r * f->t[1];
}

static void var53_parabola(flam3_iter_helper *f, double weight) {
    /* cyberxaos, 4/2007 */
    /*   r := sqrt(sqr(FTx^) + sqr(FTy^));
         FPx^ := FPx^ + parabola_height*vvar*sin(r)*sin(r)*random;  
         FPy^ := FPy^ + parabola_width*vvar*cos(r)*random; */
 
    double r = f->precalc_v_sqrt[0];
    double sr,cr;
    
    sincos(r,&sr,&cr);
    
    f->p[0] += f->xform->parabola_height * weight * sr * sr
      * flam3_random_isaac_01(f->rc);
  f->p[1] += f->xform->parabola_width * weight * cr
      * flam3_random_isaac_01(f->rc);

}

static void var54_bent2(flam3_iter_helper *f, double weight) {

   /* Bent2 in the Apophysis Plugin Pack */

   double nx = f->t[0];
  double ny = f->t[1];

   if (nx < 0.0)
      nx = nx * f->xform->bent2_x;
   if (ny < 0.0)
      ny = ny * f->xform->bent2_y;

  f->p[0] += weight * nx;
  f->p[1] += weight * ny;
}

static void var55_bipolar(flam3_iter_helper *f, double weight) {

   /* Bipolar in the Apophysis Plugin Pack */
   
   double x2y2 = f->precalc_v_sumsq[0];
   double t = x2y2+1;
  double x2 = 2 * f->t[0];
   double ps = -M_PI_2 * f->xform->bipolar_shift;
  double y = 0.5 * atan2(2.0 * f->t[1], x2y2 - 1.0) + ps;

   if (y > M_PI_2)
       y = -M_PI_2 + fmod(y + M_PI_2, M_PI);
   else if (y < -M_PI_2)
       y = M_PI_2 - fmod(M_PI_2 - y, M_PI);

  f->p[0] += weight * 0.25 * M_2_PI * log((t + x2) / (t - x2));
  f->p[1] += weight * M_2_PI * y;
}

static void var56_boarders(flam3_iter_helper *f, double weight) {

   /* Boarders in the Apophysis Plugin Pack */
   
   double roundX, roundY, offsetX, offsetY;

  roundX = rint(f->t[0]);
  roundY = rint(f->t[1]);
  offsetX = f->t[0] - roundX;
  offsetY = f->t[1] - roundY;

   if (flam3_random_isaac_01(f->rc) >= 0.75) {
    f->p[0] += weight * (offsetX * 0.5 + roundX);
    f->p[1] += weight * (offsetY * 0.5 + roundY);
   } else {

      if (fabs(offsetX) >= fabs(offsetY)) {
         
         if (offsetX >= 0.0) {
        f->p[0] += weight * (offsetX * 0.5 + roundX + 0.25);
        f->p[1] += weight * (offsetY * 0.5 + roundY + 0.25 * offsetY / offsetX);
         } else {
        f->p[0] += weight * (offsetX * 0.5 + roundX - 0.25);
        f->p[1] += weight * (offsetY * 0.5 + roundY - 0.25 * offsetY / offsetX);
         }

      } else {

         if (offsetY >= 0.0) {
        f->p[1] += weight * (offsetY * 0.5 + roundY + 0.25);
        f->p[0] += weight * (offsetX * 0.5 + roundX + offsetX / offsetY * 0.25);
         } else {
        f->p[1] += weight * (offsetY * 0.5 + roundY - 0.25);
        f->p[0] += weight * (offsetX * 0.5 + roundX - offsetX / offsetY * 0.25);
         }
      }
   }
}

static void var57_butterfly(flam3_iter_helper *f, double weight) {

   /* Butterfly in the Apophysis Plugin Pack */
   
   /* wx is weight*4/sqrt(3*pi) */
   double wx = weight*1.3029400317411197908970256609023;

  double y2 = f->t[1] * 2.0;
  double r = wx
      * sqrt(fabs(f->t[1] * f->t[0]) / (EPS + f->t[0] * f->t[0] + y2 * y2));

  f->p[0] += r * f->t[0];
  f->p[1] += r * y2;

}

static void var58_cell(flam3_iter_helper *f, double weight) {

   /* Cell in the Apophysis Plugin Pack */

   double inv_cell_size = 1.0/f->xform->cell_size;
    
   /* calculate input cell */
  int x = floor(f->t[0] * inv_cell_size);
  int y = floor(f->t[1] * inv_cell_size);

   /* Offset from cell origin */
  double dx = f->t[0] - x * f->xform->cell_size;
  double dy = f->t[1] - y * f->xform->cell_size;

   /* interleave cells */
   if (y >= 0) {
      if (x >= 0) {
         y *= 2;
         x *= 2;
      } else {
         y *= 2;
         x = -(2*x+1);
      }
   } else {
      if (x >= 0) {
         y = -(2*y+1);
         x *= 2;
      } else {
         y = -(2*y+1);
         x = -(2*x+1);
      }
   }
   
   f->p[0] += weight * (dx + x * f->xform->cell_size);
  f->p[1] -= weight * (dy + y * f->xform->cell_size);

}

static void var59_cpow(flam3_iter_helper *f, double weight) {

   /* Cpow in the Apophysis Plugin Pack */

   double a = f->precalc_atanyx;
  double lnr = 0.5 * log(f->precalc_v_sumsq[0]);
   double va = 2.0 * M_PI / f->xform->cpow_power;
   double vc = f->xform->cpow_r / f->xform->cpow_power;
   double vd = f->xform->cpow_i / f->xform->cpow_power;
   double ang = vc*a + vd*lnr + va*floor(f->xform->cpow_power*flam3_random_isaac_01(f->rc));
   double sa,ca;
   
   double m = weight * exp(vc * lnr - vd * a);
   
   sincos(ang,&sa,&ca);
   
   f->p[0] += m * ca;
  f->p[1] += m * sa;

}

static void var60_curve(flam3_iter_helper *f, double weight) {

   /* Curve in the Apophysis Plugin Pack */
   
   double pc_xlen = f->xform->curve_xlength*f->xform->curve_xlength;
   double pc_ylen = f->xform->curve_ylength*f->xform->curve_ylength;
   
   if (pc_xlen<1E-20) pc_xlen = 1E-20;
   
   if (pc_ylen<1E-20) pc_ylen = 1E-20;

  f->p[0] += weight
      * (f->t[0] + f->xform->curve_xamp * exp(-f->t[1] * f->t[1] / pc_xlen));
  f->p[1] += weight
      * (f->t[1] + f->xform->curve_yamp * exp(-f->t[0] * f->t[0] / pc_ylen));

}

static void var61_edisc(flam3_iter_helper *f, double weight) {

   /* Edisc in the Apophysis Plugin Pack */
   
   double tmp = f->precalc_v_sumsq[0] + 1.0;
  double tmp2 = 2.0 * f->t[0];
   double r1 = sqrt(tmp+tmp2);
   double r2 = sqrt(tmp-tmp2);
   double xmax = (r1+r2) * 0.5;
   double a1 = log(xmax + sqrt(xmax - 1.0));
  double a2 = -acos(f->t[0] / xmax);
   double w = weight / 11.57034632;
   double snv,csv,snhu,cshu;
   
   sincos(a1,&snv,&csv);
   
   snhu = sinh(a2);
   cshu = cosh(a2);

  if (f->t[1] > 0.0)
    snv = -snv;

  f->p[0] += w * cshu * csv;
  f->p[1] += w * snhu * snv;

}

static void var62_elliptic(flam3_iter_helper *f, double weight) {

   /* Elliptic in the Apophysis Plugin Pack */

  double tmp = f->precalc_v_sumsq[0] + 1.0;
  double x2 = 2.0 * f->t[0];
   double xmax = 0.5 * (sqrt(tmp+x2) + sqrt(tmp-x2));
  double a = f->t[0] / xmax;
   double b = 1.0 - a*a;
   double ssx = xmax - 1.0;
   double w = weight / M_PI_2;
   
   if (b<0)
      b = 0;
   else
      b = sqrt(b);
      
   if (ssx<0)
      ssx = 0;
   else
      ssx = sqrt(ssx);
      
   f->p[0] += w * atan2(a, b);

  if (f->t[1] > 0)
    f->p[1] += w * log(xmax + ssx);
   else
    f->p[1] -= w * log(xmax + ssx);

}

static void var63_escher(flam3_iter_helper *f, double weight) {

   /* Escher in the Apophysis Plugin Pack */

   double seb,ceb;
   double vc,vd;
   double m,n;
   double sn,cn;

   double a = f->precalc_atanyx;
  double lnr = 0.5 * log(f->precalc_v_sumsq[0]);

   sincos(f->xform->escher_beta,&seb,&ceb);
   
   vc = 0.5 * (1.0 + ceb);
   vd = 0.5 * seb;

   m = weight * exp(vc*lnr - vd*a);
   n = vc*a + vd*lnr;
   
   sincos(n,&sn,&cn);
   
   f->p[0] += m * cn;
  f->p[1] += m * sn;

}

static void var64_foci(flam3_iter_helper *f, double weight) {

   /* Foci in the Apophysis Plugin Pack */

  double expx = exp(f->t[0]) * 0.5;
   double expnx = 0.25 / expx;
   double sn,cn,tmp;

  sincos(f->t[1], &sn, &cn);
   tmp = weight/(expx + expnx - cn);
   
   f->p[0] += tmp * (expx - expnx);
  f->p[1] += tmp * sn;

}

static void var65_lazysusan(flam3_iter_helper *f, double weight) {

   /* Lazysusan in the Apophysis Plugin Pack */

  double x = f->t[0] - f->xform->lazysusan_x;
  double y = f->t[1] + f->xform->lazysusan_y;
   double r = sqrt(x*x + y*y);
   double sina, cosa;
   
   if (r<weight) {
      double a = atan2(y,x) + f->xform->lazysusan_spin +
                 f->xform->lazysusan_twist*(weight-r);
      sincos(a,&sina,&cosa);
      r = weight * r;
      
      f->p[0] += r * cosa + f->xform->lazysusan_x;
    f->p[1] += r * sina - f->xform->lazysusan_y;
   } else {

      r = weight * (1.0 + f->xform->lazysusan_space / r);
      
      f->p[0] += r * x + f->xform->lazysusan_x;
    f->p[1] += r * y - f->xform->lazysusan_y;

   }

}

static void var66_loonie(flam3_iter_helper *f, double weight) {

   /* Loonie in the Apophysis Plugin Pack */

   /*
    * !!! Note !!!
    * This code uses the variation weight in a non-standard fashion, and
    * it may change or even be removed in future versions of flam3.
    */
   
   double r2 = f->precalc_v_sumsq[0];
   double w2 = weight*weight;
   
   if (r2 < w2) {
      double r = weight * sqrt(w2/r2 - 1.0);
    f->p[0] += r * f->t[0];
    f->p[1] += r * f->t[1];
   } else {
    f->p[0] += weight * f->t[0];
    f->p[1] += weight * f->t[1];
   }

}

static void var67_pre_blur(flam3_iter_helper *f, double weight) {

   /* pre-xform: PreBlur (Apo 2.08) */

   /* Get pseudo-gaussian */
   double rndG = weight * (flam3_random_isaac_01(f->rc) + flam3_random_isaac_01(f->rc)
                   + flam3_random_isaac_01(f->rc) + flam3_random_isaac_01(f->rc) - 2.0);
   double rndA = flam3_random_isaac_01(f->rc) * 2.0 * M_PI;
   double sinA,cosA;
   
   sincos(rndA,&sinA,&cosA);
   
   /* Note: original coordinate changed */
  f->t[0] += rndG * cosA;
  f->t[1] += rndG * sinA;

}

static void var68_modulus(flam3_iter_helper *f, double weight) {

   /* Modulus in the Apophysis Plugin Pack */

   double xr = 2*f->xform->modulus_x;
   double yr = 2*f->xform->modulus_y;

  if (f->t[0] > f->xform->modulus_x)
    f->p[0] += weight
        * (-f->xform->modulus_x + fmod(f->t[0] + f->xform->modulus_x, xr));
  else if (f->t[0] < -f->xform->modulus_x)
    f->p[0] += weight
        * (f->xform->modulus_x - fmod(f->xform->modulus_x - f->t[0], xr));
   else
    f->p[0] += weight * f->t[0];

  if (f->t[1] > f->xform->modulus_y)
    f->p[1] += weight
        * (-f->xform->modulus_y + fmod(f->t[1] + f->xform->modulus_y, yr));
  else if (f->t[1] < -f->xform->modulus_y)
    f->p[1] += weight
        * (f->xform->modulus_y - fmod(f->xform->modulus_y - f->t[1], yr));
   else
    f->p[1] += weight * f->t[1];

}

static void var69_oscope(flam3_iter_helper *f, double weight) {

   /* oscilloscope from the apophysis plugin pack */
   
   double tpf = 2 * M_PI * f->xform->oscope_frequency;
   double t;
   
   if (f->xform->oscope_damping == 0.0)
    t = f->xform->oscope_amplitude * cos(tpf * f->t[0])
        + f->xform->oscope_separation;
   else {
    t = f->xform->oscope_amplitude
        * exp(-fabs(f->t[0]) * f->xform->oscope_damping) * cos(tpf * f->t[0])
        + f->xform->oscope_separation;
   }

  if (fabs(f->t[1]) <= t) {
    f->p[0] += weight * f->t[0];
    f->p[1] -= weight * f->t[1];
   } else {
    f->p[0] += weight * f->t[0];
    f->p[1] += weight * f->t[1];
  }
}

static void var70_polar2(flam3_iter_helper *f, double weight) {

   /* polar2 from the apophysis plugin pack */

   double p2v = weight / M_PI;
   
   f->p[0] += p2v * f->precalc_atan;
  f->p[1] += p2v / 2.0 * log(f->precalc_v_sumsq[0]);
}

static void var71_popcorn2(flam3_iter_helper *f, double weight) {

   /* popcorn2 from the apophysis plugin pack */

  f->p[0] += weight
      * (f->t[0]
          + f->xform->popcorn2_x * sin(tan(f->t[1] * f->xform->popcorn2_c)));
  f->p[1] += weight
      * (f->t[1]
          + f->xform->popcorn2_y * sin(tan(f->t[0] * f->xform->popcorn2_c)));

}

static void var72_scry(flam3_iter_helper *f, double weight) {

   /* scry from the apophysis plugin pack */
   /* note that scry does not multiply by weight, but as the */
   /* values still approach 0 as the weight approaches 0, it */
   /* should be ok                                           */ 

   /*
    * !!! Note !!!
    * This code uses the variation weight in a non-standard fashion, and
    * it may change or even be removed in future versions of flam3.
    */
   
   double t = f->precalc_v_sumsq[0];
  double r = 1.0 / (f->precalc_v_sqrt[0] * (t + 1.0 / (weight + EPS)));

   f->p[0] += f->t[0] * r;
  f->p[1] += f->t[1] * r;

}

static void var73_separation(flam3_iter_helper *f, double weight) {

   /* separation from the apophysis plugin pack */

   double sx2 = f->xform->separation_x * f->xform->separation_x;
   double sy2 = f->xform->separation_y * f->xform->separation_y;

  if (f->t[0] > 0.0)
    f->p[0] += weight
        * (sqrt(f->t[0] * f->t[0] + sx2)
            - f->t[0] * f->xform->separation_xinside);
   else
    f->p[0] -= weight
        * (sqrt(f->t[0] * f->t[0] + sx2)
            + f->t[0] * f->xform->separation_xinside);

  if (f->t[1] > 0.0)
    f->p[1] += weight
        * (sqrt(f->t[1] * f->t[1] + sy2)
            - f->t[1] * f->xform->separation_yinside);
   else
    f->p[1] -= weight
        * (sqrt(f->t[1] * f->t[1] + sy2)
            + f->t[1] * f->xform->separation_yinside);

}

static void var74_split(flam3_iter_helper *f, double weight) {

   /* Split from apo plugins pack */

  if (cos(f->t[0] * f->xform->split_xsize * M_PI) >= 0)
    f->p[1] += weight * f->t[1];
   else
    f->p[1] -= weight * f->t[1];

  if (cos(f->t[1] * f->xform->split_ysize * M_PI) >= 0)
    f->p[0] += weight * f->t[0];
   else
    f->p[0] -= weight * f->t[0];

}

static void var75_splits(flam3_iter_helper *f, double weight) {

   /* Splits from apo plugins pack */

  if (f->t[0] >= 0)
    f->p[0] += weight * (f->t[0] + f->xform->splits_x);
   else
    f->p[0] += weight * (f->t[0] - f->xform->splits_x);

  if (f->t[1] >= 0)
    f->p[1] += weight * (f->t[1] + f->xform->splits_y);
   else
    f->p[1] += weight * (f->t[1] - f->xform->splits_y);

}

static void var76_stripes(flam3_iter_helper *f, double weight) {

   /* Stripes from apo plugins pack */

   double roundx,offsetx;

  roundx = floor(f->t[0] + 0.5);
  offsetx = f->t[0] - roundx;

  f->p[0] += weight * (offsetx * (1.0 - f->xform->stripes_space) + roundx);
  f->p[1] += weight * (f->t[1] + offsetx * offsetx * f->xform->stripes_warp);

}

static void var77_wedge(flam3_iter_helper *f, double weight) {

   /* Wedge from apo plugins pack */

  double r = f->precalc_v_sqrt[0];
   double a = f->precalc_atanyx + f->xform->wedge_swirl * r;
   double c = floor( (f->xform->wedge_count * a + M_PI)*M_1_PI*0.5);
   
   double comp_fac = 1 - f->xform->wedge_angle*f->xform->wedge_count*M_1_PI*0.5;
   double sa, ca;
   
   a = a * comp_fac + c * f->xform->wedge_angle;
   
   sincos(a,&sa,&ca);

   r = weight * (r + f->xform->wedge_hole);
   
   f->p[0] += r * ca;
  f->p[1] += r * sa;

}

static void var78_wedge_julia(flam3_iter_helper *f, double weight) {

   /* wedge_julia from apo plugin pack */

  double r = weight * pow(f->precalc_v_sumsq[0], f->xform->wedgeJulia_cn);
   int t_rnd = (int)((f->xform->wedgeJulia_rN)*flam3_random_isaac_01(f->rc));
   double a = (f->precalc_atanyx + 2 * M_PI * t_rnd) / f->xform->wedge_julia_power;
   double c = floor( (f->xform->wedge_julia_count * a + M_PI)*M_1_PI*0.5 );
   double sa,ca;
   
   a = a * f->xform->wedgeJulia_cf + c * f->xform->wedge_julia_angle;
   
   sincos(a,&sa,&ca);

  f->p[0] += r * ca;
  f->p[1] += r * sa;
}

static void var79_wedge_sph(flam3_iter_helper *f, double weight) {

   /* Wedge_sph from apo plugins pack */

  double r = 1.0 / (f->precalc_v_sqrt[0] + EPS);
   double a = f->precalc_atanyx + f->xform->wedge_sph_swirl * r;
   double c = floor( (f->xform->wedge_sph_count * a + M_PI)*M_1_PI*0.5);
   
   double comp_fac = 1 - f->xform->wedge_sph_angle*f->xform->wedge_sph_count*M_1_PI*0.5;
   double sa, ca;
   
   a = a * comp_fac + c * f->xform->wedge_sph_angle;

   sincos(a,&sa,&ca);   
   r = weight * (r + f->xform->wedge_sph_hole);
   
   f->p[0] += r * ca;
  f->p[1] += r * sa;

}

static void var80_whorl(flam3_iter_helper *f, double weight) {

   /* whorl from apo plugins pack */

   /*
    * !!! Note !!!
    * This code uses the variation weight in a non-standard fashion, and
    * it may change or even be removed in future versions of flam3.
    */

  double r = f->precalc_v_sqrt[0];
   double a,sa,ca;

   if (r<weight)
      a = f->precalc_atanyx + f->xform->whorl_inside/(weight-r);
   else
      a = f->precalc_atanyx + f->xform->whorl_outside/(weight-r);

   sincos(a,&sa,&ca);

  f->p[0] += weight * r * ca;
  f->p[1] += weight * r * sa;

}

static void var81_waves2(flam3_iter_helper *f, double weight) {

   /* waves2 from Joel F */

  f->p[0] += weight
      * (f->t[0]
          + f->xform->waves2_scalex * sin(f->t[1] * f->xform->waves2_freqx));
  f->p[1] += weight
      * (f->t[1]
          + f->xform->waves2_scaley * sin(f->t[0] * f->xform->waves2_freqy));

}

/* complex vars by cothe */
/* exp log sin cos tan sec csc cot sinh cosh tanh sech csch coth */

static void var82_exp(flam3_iter_helper *f, double weight) {
   //Exponential EXP
  double expe = exp(f->t[0]);
   double expcos,expsin;
  sincos(f->t[1], &expsin, &expcos);
  f->p[0] += weight * expe * expcos;
  f->p[1] += weight * expe * expsin;
}

static void var83_log(flam3_iter_helper *f, double weight) {
   //Natural Logarithm LOG
  // needs precalc_atanyx and precalc_v_sumsq[0]
  f->p[0] += weight * 0.5 * log(f->precalc_v_sumsq[0]);
  f->p[1] += weight * f->precalc_atanyx;
}

static void var84_sin(flam3_iter_helper *f, double weight) {
   //Sine SIN
   double sinsin,sinacos,sinsinh,sincosh;
  sincos(f->t[0], &sinsin, &sinacos);
  sinsinh = sinh(f->t[1]);
  sincosh = cosh(f->t[1]);
  f->p[0] += weight * sinsin * sincosh;
  f->p[1] += weight * sinacos * sinsinh;
}

static void var85_cos(flam3_iter_helper *f, double weight) {
   //Cosine COS
   double cossin,coscos,cossinh,coscosh;
  sincos(f->t[0], &cossin, &coscos);
  cossinh = sinh(f->t[1]);
  coscosh = cosh(f->t[1]);
  f->p[0] += weight * coscos * coscosh;
  f->p[1] -= weight * cossin * cossinh;
}

static void var86_tan(flam3_iter_helper *f, double weight) {
   //Tangent TAN
   double tansin,tancos,tansinh,tancosh;
   double tanden;
  sincos(2 * f->t[0], &tansin, &tancos);
  tansinh = sinh(2.0 * f->t[1]);
  tancosh = cosh(2.0 * f->t[1]);
   tanden = 1.0/(tancos + tancosh);
  f->p[0] += weight * tanden * tansin;
  f->p[1] += weight * tanden * tansinh;
}

static void var87_sec(flam3_iter_helper *f, double weight) {
   //Secant SEC
   double secsin,seccos,secsinh,seccosh;
   double secden;
  sincos(f->t[0], &secsin, &seccos);
  secsinh = sinh(f->t[1]);
  seccosh = cosh(f->t[1]);
  secden = 2.0 / (cos(2 * f->t[0]) + cosh(2 * f->t[1]));
  f->p[0] += weight * secden * seccos * seccosh;
  f->p[1] += weight * secden * secsin * secsinh;
}

static void var88_csc(flam3_iter_helper *f, double weight) {
   //Cosecant CSC
   double cscsin,csccos,cscsinh,csccosh;
   double cscden;
  sincos(f->t[0], &cscsin, &csccos);
  cscsinh = sinh(f->t[1]);
  csccosh = cosh(f->t[1]);
  cscden = 2.0 / (cosh(2.0 * f->t[1]) - cos(2.0 * f->t[0]));
  f->p[0] += weight * cscden * cscsin * csccosh;
  f->p[1] -= weight * cscden * csccos * cscsinh;
}

static void var89_cot(flam3_iter_helper *f, double weight) {
   //Cotangent COT
   double cotsin,cotcos,cotsinh,cotcosh;
   double cotden;
  sincos(2.0 * f->t[0], &cotsin, &cotcos);
  cotsinh = sinh(2.0 * f->t[1]);
  cotcosh = cosh(2.0 * f->t[1]);
   cotden = 1.0/(cotcosh - cotcos);
  f->p[0] += weight * cotden * cotsin;
  f->p[1] += weight * cotden * -1 * cotsinh;
}

static void var90_sinh(flam3_iter_helper *f, double weight) {
   //Hyperbolic Sine SINH
   double sinhsin,sinhcos,sinhsinh,sinhcosh;
  sincos(f->t[1], &sinhsin, &sinhcos);
  sinhsinh = sinh(f->t[0]);
  sinhcosh = cosh(f->t[0]);
  f->p[0] += weight * sinhsinh * sinhcos;
  f->p[1] += weight * sinhcosh * sinhsin;
}

static void var91_cosh(flam3_iter_helper *f, double weight) {
   //Hyperbolic Cosine COSH
   double coshsin,coshcos,coshsinh,coshcosh;
  sincos(f->t[1], &coshsin, &coshcos);
  coshsinh = sinh(f->t[0]);
  coshcosh = cosh(f->t[0]);
  f->p[0] += weight * coshcosh * coshcos;
  f->p[1] += weight * coshsinh * coshsin;
}

static void var92_tanh(flam3_iter_helper *f, double weight) {
   //Hyperbolic Tangent TANH
   double tanhsin,tanhcos,tanhsinh,tanhcosh;
   double tanhden;
  sincos(2.0 * f->t[1], &tanhsin, &tanhcos);
  tanhsinh = sinh(2.0 * f->t[0]);
  tanhcosh = cosh(2.0 * f->t[0]);
   tanhden = 1.0/(tanhcos + tanhcosh);
  f->p[0] += weight * tanhden * tanhsinh;
  f->p[1] += weight * tanhden * tanhsin;
}

static void var93_sech(flam3_iter_helper *f, double weight) {
   //Hyperbolic Secant SECH
   double sechsin,sechcos,sechsinh,sechcosh;
   double sechden;
  sincos(f->t[1], &sechsin, &sechcos);
  sechsinh = sinh(f->t[0]);
  sechcosh = cosh(f->t[0]);
  sechden = 2.0 / (cos(2.0 * f->t[1]) + cosh(2.0 * f->t[0]));
  f->p[0] += weight * sechden * sechcos * sechcosh;
  f->p[1] -= weight * sechden * sechsin * sechsinh;
}

static void var94_csch(flam3_iter_helper *f, double weight) {
   //Hyperbolic Cosecant CSCH
   double cschsin,cschcos,cschsinh,cschcosh;
   double cschden;
  sincos(f->t[1], &cschsin, &cschcos);
  cschsinh = sinh(f->t[0]);
  cschcosh = cosh(f->t[0]);
  cschden = 2.0 / (cosh(2.0 * f->t[0]) - cos(2.0 * f->t[1]));
  f->p[0] += weight * cschden * cschsinh * cschcos;
  f->p[1] -= weight * cschden * cschcosh * cschsin;
}

static void var95_coth(flam3_iter_helper *f, double weight) {
   //Hyperbolic Cotangent COTH
   double cothsin,cothcos,cothsinh,cothcosh;
   double cothden;
  sincos(2.0 * f->t[1], &cothsin, &cothcos);
  cothsinh = sinh(2.0 * f->t[0]);
  cothcosh = cosh(2.0 * f->t[0]);
   cothden = 1.0/(cothcosh - cothcos);
  f->p[0] += weight * cothden * cothsinh;
  f->p[1] += weight * cothden * cothsin;
}

static void var96_auger(flam3_iter_helper *f, double weight) {

    // Auger, by Xyrus01
  double s = sin(f->xform->auger_freq * f->t[0]);
  double t = sin(f->xform->auger_freq * f->t[1]);
  double dy = f->t[1]
      + f->xform->auger_weight
          * (f->xform->auger_scale * s / 2.0 + fabs(f->t[1]) * s);
  double dx = f->t[0]
      + f->xform->auger_weight
          * (f->xform->auger_scale * t / 2.0 + fabs(f->t[0]) * t);

  f->p[0] += weight * (f->t[0] + f->xform->auger_sym * (dx - f->t[0]));
  f->p[1] += weight * dy;
}

static void var97_flux(flam3_iter_helper *f, double weight) {

    // Flux, by meckie
  double xpw = f->t[0] + weight;
  double xmw = f->t[0] - weight;
  double avgr = weight * (2 + f->xform->flux_spread)
      * sqrt(
          sqrt(f->t[1] * f->t[1] + xpw * xpw)
              / sqrt(f->t[1] * f->t[1] + xmw * xmw));
  double avga = (atan2(f->t[1], xmw) - atan2(f->t[1], xpw)) * 0.5;

  f->p[0] += avgr * cos(avga);
  f->p[1] += avgr * sin(avga);
}

static void var98_mobius(flam3_iter_helper *f, double weight) {

    // Mobius, by eralex
    double re_u, im_u, re_v, im_v, rad_v;

  re_u = f->xform->mobius_re_a * f->t[0] - f->xform->mobius_im_a * f->t[1]
      + f->xform->mobius_re_b;
  im_u = f->xform->mobius_re_a * f->t[1] + f->xform->mobius_im_a * f->t[0]
      + f->xform->mobius_im_b;
  re_v = f->xform->mobius_re_c * f->t[0] - f->xform->mobius_im_c * f->t[1]
      + f->xform->mobius_re_d;
  im_v = f->xform->mobius_re_c * f->t[1] + f->xform->mobius_im_c * f->t[0]
      + f->xform->mobius_im_d;

    rad_v = weight / (re_v*re_v + im_v*im_v);

  f->p[0] += rad_v * (re_u * re_v + im_u * im_v);
  f->p[1] += rad_v * (im_u * re_v - re_u * im_v);
}


/* Precalc functions */

static void perspective_precalc(flam3_xform *xf) {
   double ang = xf->perspective_angle * M_PI / 2.0;
   xf->persp_vsin = sin(ang);
   xf->persp_vfcos = xf->perspective_dist * cos(ang);
}

static void juliaN_precalc(flam3_xform *xf) {
   xf->julian_rN = fabs(xf->julian_power);
   xf->julian_cn = xf->julian_dist / (double)xf->julian_power / 2.0;
}

static void wedgeJulia_precalc(flam3_xform *xf) {
   xf->wedgeJulia_cf = 1.0 - xf->wedge_julia_angle * xf->wedge_julia_count * M_1_PI * 0.5;
   xf->wedgeJulia_rN = fabs(xf->wedge_julia_power);
   xf->wedgeJulia_cn = xf->wedge_julia_dist / xf->wedge_julia_power / 2.0;
}

static void juliaScope_precalc(flam3_xform *xf) {
   xf->juliascope_rN = fabs(xf->juliascope_power);
   xf->juliascope_cn = xf->juliascope_dist / (double)xf->juliascope_power / 2.0;
}

static void radial_blur_precalc(flam3_xform *xf) {
   sincos(xf->radial_blur_angle * M_PI / 2.0,
             &xf->radialBlur_spinvar, &xf->radialBlur_zoomvar);
}

static void waves_precalc(flam3_xform *xf) {
   double dx = xf->c[2][0];
   double dy = xf->c[2][1];

   xf->waves_dx2 = 1.0/(dx * dx + EPS);
   xf->waves_dy2 = 1.0/(dy * dy + EPS);
}

static void disc2_precalc(flam3_xform *xf) {
   double add = xf->disc2_twist;
   double k;

   xf->disc2_timespi = xf->disc2_rot * M_PI;

   sincos(add,&xf->disc2_sinadd,&xf->disc2_cosadd);
   xf->disc2_cosadd -= 1;

   if (add > 2 * M_PI) {
      k = (1 + add - 2*M_PI);
      xf->disc2_cosadd *= k;
      xf->disc2_sinadd *= k;
   }

   if (add < -2 * M_PI) {
      k = (1 + add + 2*M_PI);
      xf->disc2_cosadd *= k;
      xf->disc2_sinadd *= k;
   }
}

static void supershape_precalc(flam3_xform *xf) {
   xf->super_shape_pm_4 = xf->super_shape_m / 4.0;
   xf->super_shape_pneg1_n1 = -1.0 / xf->super_shape_n1;
}

void xform_precalc(flam3_genome *cp, int xi) {

   perspective_precalc(&(cp->xform[xi]));
   juliaN_precalc(&(cp->xform[xi]));
   juliaScope_precalc(&(cp->xform[xi]));
   radial_blur_precalc(&(cp->xform[xi]));
   waves_precalc(&(cp->xform[xi]));
   disc2_precalc(&(cp->xform[xi]));
   supershape_precalc(&(cp->xform[xi]));
   wedgeJulia_precalc(&(cp->xform[xi]));   
}   

typedef void (*varFuncPtr)(flam3_iter_helper *f, double weight);

static varFuncPtr varFuncTab[flam3_nvariations] = {
   &var0_linear,
   &var1_sinusoidal,
   &var2_spherical,
   &var3_swirl,
   &var4_horseshoe,
   &var5_polar,
   &var6_handkerchief,
   &var7_heart,
   &var8_disc,
   &var9_spiral,
   &var10_hyperbolic,
   &var11_diamond,
   &var12_ex,
   &var13_julia,
   &var14_bent,
   &var15_waves,
   &var16_fisheye,
   &var17_popcorn,
   &var18_exponential,
   &var19_power,
   &var20_cosine,
   &var21_rings,
   &var22_fan,
   &var23_blob,
   &var24_pdj,
   &var25_fan2,
   &var26_rings2,
   &var27_eyefish,
   &var28_bubble,
   &var29_cylinder,
   &var30_perspective,
   &var31_noise,
   &var32_juliaN_generic,
   &var33_juliaScope_generic,
   &var34_blur,
   &var35_gaussian,
   &var36_radial_blur,
   &var37_pie,
   &var38_ngon,
   &var39_curl,
   &var40_rectangles,
   &var41_arch,
   &var42_tangent,
   &var43_square,
   &var44_rays,
   &var45_blade,
   &var46_secant2,
   &var47_twintrian,
   &var48_cross,
   &var49_disc2,
   &var50_supershape,
   &var51_flower,
   &var52_conic,
   &var53_parabola,
   &var54_bent2,
   &var55_bipolar,
   &var56_boarders,
   &var57_butterfly,
   &var58_cell,
   &var59_cpow,
   &var60_curve,
   &var61_edisc,
   &var62_elliptic,
   &var63_escher,
   &var64_foci,
   &var65_lazysusan,
   &var66_loonie,
   NULL,
   &var68_modulus,
   &var69_oscope,
   &var70_polar2,
   &var71_popcorn2,
   &var72_scry,
   &var73_separation,
   &var74_split,
   &var75_splits,
   &var76_stripes,
   &var77_wedge,
   &var78_wedge_julia,
   &var79_wedge_sph,
   &var80_whorl,
   &var81_waves2,
   &var82_exp,
   &var83_log,
   &var84_sin,
   &var85_cos,
   &var86_tan,
   &var87_sec,
   &var88_csc,
   &var89_cot,
   &var90_sinh,
   &var91_cosh,
   &var92_tanh,
   &var93_sech,
   &var94_csch,
   &var95_coth,
   &var96_auger,
   &var97_flux,
   &var98_mobius
};

int prepare_precalc_flags(flam3_genome *cp) {

   double d;
   int i,j,totnum;

   /* Loop over valid xforms */
   for (i = 0; i < cp->num_xforms; i++) {
      d = cp->xform[i].density;
      if (d < 0.0) {
         fprintf(stderr, "xform %d weight must be non-negative, not %g.\n",i,d);
         return(1);
      }

      if (i != cp->final_xform_index && d == 0.0)
         continue;

      totnum = 0;

      cp->xform[i].vis_adjusted = adjust_percentage(cp->xform[i].opacity);

      cp->xform[i].precalc_angles_flag=0;
      cp->xform[i].precalc_atan_xy_flag=0;
      cp->xform[i].precalc_atan_yx_flag=0;
      cp->xform[i].has_preblur=0;
      cp->xform[i].has_post = !(id_matrix(cp->xform[i].post));


      for (j = 0; j < flam3_nvariations; j++) {

         if (cp->xform[i].var[j]!=0) {

            cp->xform[i].varFunc[totnum] = (void*)varFuncTab[j];
            cp->xform[i].active_var_weights[totnum] = cp->xform[i].var[j];

            if (j==VAR_POLAR) {
               cp->xform[i].precalc_atan_xy_flag=1;
            } else if (j==VAR_HANDKERCHIEF) {
               cp->xform[i].precalc_atan_xy_flag=1;
            } else if (j==VAR_HEART) {
               cp->xform[i].precalc_atan_xy_flag=1;
            } else if (j==VAR_DISC) {
               cp->xform[i].precalc_atan_xy_flag=1;
            } else if (j==VAR_SPIRAL) {
               cp->xform[i].precalc_angles_flag=1;
            } else if (j==VAR_HYPERBOLIC) {
               cp->xform[i].precalc_angles_flag=1;
            } else if (j==VAR_DIAMOND) {
               cp->xform[i].precalc_angles_flag=1;
            } else if (j==VAR_EX) {
               cp->xform[i].precalc_atan_xy_flag=1;
            } else if (j==VAR_JULIA) {
               cp->xform[i].precalc_atan_xy_flag=1;
            } else if (j==VAR_POWER) {
               cp->xform[i].precalc_angles_flag=1;
            } else if (j==VAR_RINGS) {
               cp->xform[i].precalc_angles_flag=1;
            } else if (j==VAR_FAN) {
               cp->xform[i].precalc_atan_xy_flag=1;
            } else if (j==VAR_BLOB) {
               cp->xform[i].precalc_atan_xy_flag=1;
               cp->xform[i].precalc_angles_flag=1;
            } else if (j==VAR_FAN2) {
               cp->xform[i].precalc_atan_xy_flag=1;
            } else if (j==VAR_RINGS2) {
               cp->xform[i].precalc_angles_flag=1;
            } else if (j==VAR_JULIAN) {
               cp->xform[i].precalc_atan_yx_flag=1;
            } else if (j==VAR_JULIASCOPE) {
               cp->xform[i].precalc_atan_yx_flag=1;
            } else if (j==VAR_RADIAL_BLUR) {
               cp->xform[i].precalc_atan_yx_flag=1;
            } else if (j==VAR_NGON) {
               cp->xform[i].precalc_atan_yx_flag=1;
            } else if (j==VAR_DISC2) {
               cp->xform[i].precalc_atan_xy_flag=1;
            } else if (j==VAR_SUPER_SHAPE) {
               cp->xform[i].precalc_atan_yx_flag=1;
            } else if (j==VAR_FLOWER) {
               cp->xform[i].precalc_atan_yx_flag=1;
            } else if (j==VAR_CONIC) {
               cp->xform[i].precalc_atan_yx_flag=1;
            } else if (j==VAR_CPOW) {
               cp->xform[i].precalc_atan_yx_flag=1;
            } else if (j==VAR_ESCHER) {
               cp->xform[i].precalc_atan_yx_flag=1;
            } else if (j==VAR_PRE_BLUR) {
               cp->xform[i].has_preblur=cp->xform[i].var[j];
            } else if (j==VAR_POLAR2) {
               cp->xform[i].precalc_atan_xy_flag=1;
            } else if (j==VAR_WEDGE) {
               cp->xform[i].precalc_atan_yx_flag=1;
            } else if (j==VAR_WEDGE_JULIA) {
               cp->xform[i].precalc_atan_yx_flag=1;
            } else if (j==VAR_WEDGE_SPH) {
               cp->xform[i].precalc_atan_yx_flag=1;
            } else if (j==VAR_WHORL) {
               cp->xform[i].precalc_atan_yx_flag=1;
            } else if (j==VAR_LOG) {
               cp->xform[i].precalc_atan_yx_flag=1;
            }
            
            totnum++;
         }
      }

      cp->xform[i].num_active_vars = totnum;

   }
   
   return(0);
}

static __m128d apply_affine(__m128d p, v2d *c) {
  const __m128d v_off = _mm_load_pd(c[2]);

  const __m128d v_R0t = _mm_load_pd(c[0]);
  const __m128d v_R1t = _mm_load_pd(c[1]);
  const __m128d v_R0 = _mm_unpacklo_pd(v_R0t, v_R1t);
  const __m128d v_R1 = _mm_unpackhi_pd(v_R0t, v_R1t);

  const __m128d v_p0t = _mm_mul_pd(p, v_R0);
  const __m128d v_p1t = _mm_mul_pd(p, v_R1);
  const __m128d v_p = _mm_hadd_pd(v_p0t, v_p1t);
  return _mm_add_pd(v_p, v_off);
}

__m256d apply_xform(flam3_genome * const cp, const int fn, const __m256d p,
    randctx * const rc, int * const badvals, int consec)
{
  flam3_iter_helper f;
  int var_n;

  __m128d q10;

  f.rc = rc;

  const double s1 = cp->xform[fn].color_speed;

  const double q2 = s1 * cp->xform[fn].color + (1.0 - s1) * p[2];

  const __m128d t = apply_affine(_mm256_extractf128_pd(p, 0), cp->xform[fn].c);

  /* Always calculate sumsq and sqrt */
  __m128d v_t2 = _mm_mul_pd(t, t);
  __m128d v_r2 = _mm_hadd_pd(v_t2, v_t2);
  __m128d v_r = _mm_sqrt_pd(v_r2);
  f.precalc_v_sumsq = v_r2;
  f.precalc_v_sqrt = v_r;

  /* Check to see if we can precalculate any parts */
  /* Precalculate atanxy, sin, cos */
  if (cp->xform[fn].precalc_atan_xy_flag > 0) {
    f.precalc_atan = atan2(t[0], t[1]);
  }

  if (cp->xform[fn].precalc_angles_flag > 0) {
    f.precalc_sina = t[0] / f.precalc_v_sqrt[0];
    f.precalc_cosa = t[1] / f.precalc_v_sqrt[0];
  }

  /* Precalc atanyx */
  if (cp->xform[fn].precalc_atan_yx_flag > 0) {
    f.precalc_atanyx = atan2(t[1], t[0]);
  }

  f.t = t;
  f.p = _mm_setzero_pd();
  f.xform = &(cp->xform[fn]);

  for (var_n = 0; var_n < cp->xform[fn].num_active_vars; var_n++) {
    const double weight = cp->xform[fn].active_var_weights[var_n];
    varFuncPtr varFunc = (varFuncPtr) (cp->xform[fn].varFunc[var_n]);
    varFunc(&f, weight);
  }

  /* apply the post transform */
  if (cp->xform[fn].has_post) {
    q10 = apply_affine(f.p, cp->xform[fn].post);
  } else {
    q10 = f.p;
  }

  __m256d q = _mm256_set_pd(q2, q2, q10[1], q10[0]);

  /* Check for badvalues and return randoms if bad */
  if (badvalue(q[0]) || badvalue(q[1])) {
    q[0] = flam3_random_isaac_11(rc);
    q[1] = flam3_random_isaac_11(rc);
    *badvals += 1;

    consec++;
    if (consec < 5) {
      return apply_xform(cp, fn, q, rc, badvals, consec);
    }
  }

  return q;
}

void initialize_xforms(flam3_genome *thiscp, int start_here) {

   int i,j;
   for (i = start_here ; i < thiscp->num_xforms ; i++) {
      thiscp->xform[i].padding = 0;
      thiscp->xform[i].density = 0.0;
      thiscp->xform[i].color_speed = 0.5;
      thiscp->xform[i].animate = 1.0;
      thiscp->xform[i].color = i&1;
      thiscp->xform[i].opacity = 1.0;
      thiscp->xform[i].var[0] = 1.0;
      thiscp->xform[i].motion_freq = 0;
      thiscp->xform[i].motion_func = 0;
      thiscp->xform[i].num_motion = 0;
      thiscp->xform[i].motion = NULL;
      for (j = 1; j < flam3_nvariations; j++)
         thiscp->xform[i].var[j] = 0.0;
      thiscp->xform[i].c[0][0] = 1.0;
      thiscp->xform[i].c[0][1] = 0.0;
      thiscp->xform[i].c[1][0] = 0.0;
      thiscp->xform[i].c[1][1] = 1.0;
      thiscp->xform[i].c[2][0] = 0.0;
      thiscp->xform[i].c[2][1] = 0.0;
      thiscp->xform[i].post[0][0] = 1.0;
      thiscp->xform[i].post[0][1] = 0.0;
      thiscp->xform[i].post[1][0] = 0.0;
      thiscp->xform[i].post[1][1] = 1.0;
      thiscp->xform[i].post[2][0] = 0.0;
      thiscp->xform[i].post[2][1] = 0.0;
      thiscp->xform[i].wind[0] = 0.0;
      thiscp->xform[i].wind[1] = 0.0;
      thiscp->xform[i].blob_low = 0.0;
      thiscp->xform[i].blob_high = 1.0;
      thiscp->xform[i].blob_waves = 1.0;
      thiscp->xform[i].pdj_a = 0.0;
      thiscp->xform[i].pdj_b = 0.0;
      thiscp->xform[i].pdj_c = 0.0;
      thiscp->xform[i].pdj_d = 0.0;
      thiscp->xform[i].fan2_x = 0.0;
      thiscp->xform[i].fan2_y = 0.0;
      thiscp->xform[i].rings2_val = 0.0;
      thiscp->xform[i].perspective_angle = 0.0;
      thiscp->xform[i].perspective_dist = 0.0;
      thiscp->xform[i].persp_vsin = 0.0;
      thiscp->xform[i].persp_vfcos = 0.0;
      thiscp->xform[i].radial_blur_angle = 0.0;
      thiscp->xform[i].disc2_rot = 0.0;
      thiscp->xform[i].disc2_twist = 0.0;
      thiscp->xform[i].disc2_sinadd = 0.0;
      thiscp->xform[i].disc2_cosadd = 0.0;
      thiscp->xform[i].disc2_timespi = 0.0;
      thiscp->xform[i].flower_petals = 0.0;
      thiscp->xform[i].flower_holes = 0.0;
      thiscp->xform[i].parabola_height = 0.0;
      thiscp->xform[i].parabola_width = 0.0;
      thiscp->xform[i].bent2_x = 1.0;
      thiscp->xform[i].bent2_y = 1.0;
      thiscp->xform[i].bipolar_shift = 0.0;
      thiscp->xform[i].cell_size = 1.0;
      thiscp->xform[i].cpow_r = 1.0;
      thiscp->xform[i].cpow_i = 0.0;
      thiscp->xform[i].cpow_power = 1.0;
      thiscp->xform[i].curve_xamp = 0.0;
      thiscp->xform[i].curve_yamp = 0.0;
      thiscp->xform[i].curve_xlength = 1.0;
      thiscp->xform[i].curve_ylength = 1.0;
      thiscp->xform[i].escher_beta = 0.0;
      thiscp->xform[i].lazysusan_space = 0.0;
      thiscp->xform[i].lazysusan_twist = 0.0;
      thiscp->xform[i].lazysusan_spin = 0.0;
      thiscp->xform[i].lazysusan_x = 0.0;
      thiscp->xform[i].lazysusan_y = 0.0;
      thiscp->xform[i].modulus_x = 0.0;
      thiscp->xform[i].modulus_y = 0.0;
      thiscp->xform[i].oscope_separation = 1.0;
      thiscp->xform[i].oscope_frequency = M_PI;
      thiscp->xform[i].oscope_amplitude = 1.0;
      thiscp->xform[i].oscope_damping = 0.0;
      thiscp->xform[i].popcorn2_c = 0.0;
      thiscp->xform[i].popcorn2_x = 0.0;
      thiscp->xform[i].popcorn2_y = 0.0;
      thiscp->xform[i].separation_x = 0.0;
      thiscp->xform[i].separation_xinside = 0.0;
      thiscp->xform[i].separation_y = 0.0;
      thiscp->xform[i].separation_yinside = 0.0;
      thiscp->xform[i].split_xsize = 0.0;
      thiscp->xform[i].split_ysize = 0.0;
      thiscp->xform[i].splits_x = 0.0;
      thiscp->xform[i].splits_y = 0.0;
      thiscp->xform[i].stripes_space = 0.0;
      thiscp->xform[i].stripes_warp = 0.0;
      thiscp->xform[i].wedge_angle = 0.0;
      thiscp->xform[i].wedge_hole = 0.0;
      thiscp->xform[i].wedge_count = 1.0;
      thiscp->xform[i].wedge_swirl = 0.0;
      thiscp->xform[i].wedge_sph_angle = 0.0;
      thiscp->xform[i].wedge_sph_hole = 0.0;
      thiscp->xform[i].wedge_sph_count = 1.0;
      thiscp->xform[i].wedge_sph_swirl = 0.0;

      thiscp->xform[i].wedge_julia_power = 1.0;
      thiscp->xform[i].wedge_julia_dist = 0.0;
      thiscp->xform[i].wedge_julia_count = 1.0;
      thiscp->xform[i].wedge_julia_angle = 0.0;
      thiscp->xform[i].wedgeJulia_cf = 0.0;
      thiscp->xform[i].wedgeJulia_cn = 0.5;
      thiscp->xform[i].wedgeJulia_rN = 1.0;
      thiscp->xform[i].whorl_inside = 0.0;
      thiscp->xform[i].whorl_outside = 0.0;
      
      thiscp->xform[i].waves2_scalex = 0.0;       
      thiscp->xform[i].waves2_scaley = 0.0;       
      thiscp->xform[i].waves2_freqx = 0.0;       
      thiscp->xform[i].waves2_freqy = 0.0;  
      
      thiscp->xform[i].auger_freq = 1.0;
      thiscp->xform[i].auger_weight = 0.5;
      thiscp->xform[i].auger_sym = 0.0;
      thiscp->xform[i].auger_scale = 1.0;     

      thiscp->xform[i].flux_spread = 0.0;
       
      thiscp->xform[i].julian_power = 1.0;
      thiscp->xform[i].julian_dist = 1.0;
      thiscp->xform[i].julian_rN = 1.0;
      thiscp->xform[i].julian_cn = 0.5;
      thiscp->xform[i].juliascope_power = 1.0;
      thiscp->xform[i].juliascope_dist = 1.0;
      thiscp->xform[i].juliascope_rN = 1.0;
      thiscp->xform[i].juliascope_cn = 0.5;
      thiscp->xform[i].radialBlur_spinvar = 0.0;
      thiscp->xform[i].radialBlur_zoomvar = 1.0;
      thiscp->xform[i].pie_slices = 6.0;
      thiscp->xform[i].pie_rotation = 0.0;
      thiscp->xform[i].pie_thickness = 0.5;
      thiscp->xform[i].ngon_sides = 5;
      thiscp->xform[i].ngon_power = 3;
      thiscp->xform[i].ngon_circle = 1;
      thiscp->xform[i].ngon_corners = 2;
      thiscp->xform[i].curl_c1 = 1.0;
      thiscp->xform[i].curl_c2 = 0.0;
      thiscp->xform[i].rectangles_x = 1.0;
      thiscp->xform[i].rectangles_y = 1.0;
      thiscp->xform[i].amw_amp = 1.0;
      thiscp->xform[i].super_shape_rnd = 0.0;
      thiscp->xform[i].super_shape_m = 0.0;
      thiscp->xform[i].super_shape_n1 = 1.0;
      thiscp->xform[i].super_shape_n2 = 1.0;
      thiscp->xform[i].super_shape_n3 = 1.0;
      thiscp->xform[i].super_shape_holes = 0.0;
      thiscp->xform[i].conic_eccentricity = 1.0;
      thiscp->xform[i].conic_holes = 0.0;

      thiscp->xform[i].mobius_re_a = 0.0;
      thiscp->xform[i].mobius_re_b = 0.0;
      thiscp->xform[i].mobius_re_c = 0.0;
      thiscp->xform[i].mobius_re_d = 0.0;
      thiscp->xform[i].mobius_im_a = 0.0;
      thiscp->xform[i].mobius_im_b = 0.0;
      thiscp->xform[i].mobius_im_c = 0.0;
      thiscp->xform[i].mobius_im_d = 0.0;
   }
}
