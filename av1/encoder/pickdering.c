/*
 * Copyright (c) 2016, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include <string.h>
#include <math.h>

#include "./aom_scale_rtcd.h"
#include "av1/common/dering.h"
#include "av1/common/onyxc_int.h"
#include "av1/common/reconinter.h"
#include "av1/encoder/encoder.h"
#include "aom/aom_integer.h"

static double compute_dist(int16_t *x, int xstride, int16_t *y, int ystride,
                           int nhb, int nvb, int coeff_shift) {
  int i, j;
  double sum;
  sum = 0;
  for (i = 0; i < nvb << 3; i++) {
    for (j = 0; j < nhb << 3; j++) {
      double tmp;
      tmp = x[i * xstride + j] - y[i * ystride + j];
      sum += tmp * tmp;
    }
  }
  return sum / (double)(1 << 2 * coeff_shift);
}

#if 1
static int64_t compute_dist_wrap(const uint8_t *src, int src_stride,
                              const uint8_t *dst, int dst_stride, int bsw, int bsh,
                              int coeff_shift) {
  int i, j;
  int64_t d;
  DECLARE_ALIGNED(16, od_coeff, orig[MAX_TX_SQUARE]);
  DECLARE_ALIGNED(16, od_coeff, rec[MAX_TX_SQUARE]);

  for (j = 0; j < bsh; j++)
    for (i = 0; i < bsw; i++) orig[j * bsw + i] = src[j * src_stride + i];

  for (j = 0; j < bsh; j++)
    for (i = 0; i < bsw; i++) rec[j * bsw + i] = dst[j * dst_stride + i];

  d = (int64_t)od_compute_dist(qm, 1, orig, rec, bsw, bsh,
                               0);
  return d;
}
#else
static int od_compute_var_4x4(od_coeff *x, int stride) {
  int sum;
  int s2;
  int i;
  sum = 0;
  s2 = 0;
  for (i = 0; i < 4; i++) {
    int j;
    for (j = 0; j < 4; j++) {
      int t;

      t = x[i * stride + j];
      sum += t;
      s2 += t * t;
    }
  }
  // TODO(yushin) : Check wheter any changes are required for high bit depth.
  return (s2 - (sum * sum >> 4)) >> 4;
}

/* OD_DIST_LP_MID controls the frequency weighting filter used for computing
   the distortion. For a value X, the filter is [1 X 1]/(X + 2) and
   is applied both horizontally and vertically. For X=5, the filter is
   a good approximation for the OD_QM8_Q4_HVS quantization matrix. */
#define OD_DIST_LP_MID (5)
#define OD_DIST_LP_NORM (OD_DIST_LP_MID + 2)

static double od_compute_dist_8x8(od_coeff *x,
                                  od_coeff *y, od_coeff *e_lp, int stride) {
  double sum;
  int min_var;
  double mean_var;
  double var_stat;
  double activity;
  double calibration;
  int i;
  int j;
  double vardist;

  vardist = 0;
#if 1
  min_var = INT_MAX;
  mean_var = 0;
  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      int varx;
      int vary;
      varx = od_compute_var_4x4(x + 2 * i * stride + 2 * j, stride);
      vary = od_compute_var_4x4(y + 2 * i * stride + 2 * j, stride);
      min_var = OD_MINI(min_var, 0);
      mean_var += 1. / (1 + 0);
      /* The cast to (double) is to avoid an overflow before the sqrt.*/
      vardist += varx - 2 * sqrt(varx * (double)vary) + vary;
    }
  }
  /* We use a different variance statistic depending on whether activity
     masking is used, since the harmonic mean appeared slghtly worse with
     masking off. The calibration constant just ensures that we preserve the
     rate compared to activity=1. */
  calibration = 1.95;
  var_stat = 9. / mean_var;
  /* 1.62 is a calibration constant, 0.25 is a noise floor and 1/6 is the
     activity masking constant. */
  activity = calibration * pow(.25 + var_stat, -1. / 6);
#else
  activity = 1;
#endif
  sum = 0;
  for (i = 0; i < 8; i++) {
    for (j = 0; j < 8; j++)
      sum += e_lp[i * stride + j] * (double)e_lp[i * stride + j];
  }
  /* Normalize the filter to unit DC response. */
  sum *= 1. / (OD_DIST_LP_NORM * OD_DIST_LP_NORM * OD_DIST_LP_NORM *
               OD_DIST_LP_NORM);
  return activity * activity * (sum);
}

// Note : Inputs x and y are in a pixel domain
static double od_compute_dist(od_coeff *x,
                              od_coeff *y, int bsize_w, int bsize_h,
                              int qindex) {
  int i;
  double sum;
  sum = 0;

  (void)qindex;

  assert(bsize_w >= 8 && bsize_h >= 8);

    int j;
    DECLARE_ALIGNED(16, od_coeff, e[MAX_TX_SQUARE]);
    DECLARE_ALIGNED(16, od_coeff, tmp[MAX_TX_SQUARE]);
    DECLARE_ALIGNED(16, od_coeff, e_lp[MAX_TX_SQUARE]);
    int mid = OD_DIST_LP_MID;
    for (i = 0; i < bsize_h; i++) {
      for (j = 0; j < bsize_w; j++) {
        e[i * bsize_w + j] = x[i * bsize_w + j] - y[i * bsize_w + j];
      }
    }
    for (i = 0; i < bsize_h; i++) {
      tmp[i * bsize_w] = mid * e[i * bsize_w] + 2 * e[i * bsize_w + 1];
      tmp[i * bsize_w + bsize_w - 1] =
          mid * e[i * bsize_w + bsize_w - 1] + 2 * e[i * bsize_w + bsize_w - 2];
      for (j = 1; j < bsize_w - 1; j++) {
        tmp[i * bsize_w + j] = mid * e[i * bsize_w + j] +
                               e[i * bsize_w + j - 1] + e[i * bsize_w + j + 1];
      }
    }
    for (j = 0; j < bsize_w; j++) {
      e_lp[j] = mid * tmp[j] + 2 * tmp[bsize_w + j];
      e_lp[(bsize_h - 1) * bsize_w + j] =
          mid * tmp[(bsize_h - 1) * bsize_w + j] +
          2 * tmp[(bsize_h - 2) * bsize_w + j];
    }
    for (i = 1; i < bsize_h - 1; i++) {
      for (j = 0; j < bsize_w; j++) {
        e_lp[i * bsize_w + j] = mid * tmp[i * bsize_w + j] +
                                tmp[(i - 1) * bsize_w + j] +
                                tmp[(i + 1) * bsize_w + j];
      }
    }
    for (i = 0; i < bsize_h; i += 8) {
      for (j = 0; j < bsize_w; j += 8) {
        sum += od_compute_dist_8x8(&x[i * bsize_w + j],
                                   &y[i * bsize_w + j], &e_lp[i * bsize_w + j],
                                   bsize_w);
      }
    }
    /* Compensate for the fact that the quantization matrix lowers the
       distortion value. We tried a half-dozen values and picked the one where
       we liked the ntt-short1 curves best. The tuning is approximate since
       the different metrics go in different directions. */
    /*Start interpolation at coded_quantizer 1.7=f(36) and end it at 1.2=f(47)*/
    // TODO(yushin): Check whether qindex of AV1 work here, replacing daala's
    // coded_quantizer.
    /*sum *= qindex >= 47 ? 1.2 :
        qindex <= 36 ? 1.7 :
     1.7 + (1.2 - 1.7)*(qindex - 36)/(47 - 36);*/
  return sum;
}

static double compute_dist_wrap(const int16_t *_x, int xstride, const int16_t *_y, int ystride,
                           int nhb, int nvb, int coeff_shift) {
  od_coeff x[MAX_TX_SQUARE];
  od_coeff y[MAX_TX_SQUARE];
  int i, j;
  for (i=0;i<nvb*8;i++) {
    for (j=0;j<nhb*8;j++) {
      x[i*nvb*8 + j] = _x[i*xstride + j];
      y[i*nvb*8 + j] = _y[i*ystride + j];
    }
  }
  return od_compute_dist(x, y, 8*nhb, 8*nvb, 0);
}
#endif

int av1_dering_search(YV12_BUFFER_CONFIG *frame, const YV12_BUFFER_CONFIG *ref,
                      AV1_COMMON *cm, MACROBLOCKD *xd) {
  int r, c;
  int sbr, sbc;
  int nhsb, nvsb;
  int16_t *src;
  int16_t *ref_coeff;
  dering_list dlist[MAX_MIB_SIZE * MAX_MIB_SIZE];
  int dir[OD_DERING_NBLOCKS][OD_DERING_NBLOCKS] = { { 0 } };
  int stride;
  int bsize[3];
  int dec[3];
  int pli;
  int level;
  int best_level;
  int dering_count;
  int coeff_shift = AOMMAX(cm->bit_depth - 8, 0);
  src = aom_malloc(sizeof(*src) * cm->mi_rows * cm->mi_cols * 64);
  ref_coeff = aom_malloc(sizeof(*ref_coeff) * cm->mi_rows * cm->mi_cols * 64);
  av1_setup_dst_planes(xd->plane, frame, 0, 0);
  for (pli = 0; pli < 3; pli++) {
    dec[pli] = xd->plane[pli].subsampling_x;
    bsize[pli] = OD_DERING_SIZE_LOG2 - dec[pli];
  }
  stride = cm->mi_cols << bsize[0];
  for (r = 0; r < cm->mi_rows << bsize[0]; ++r) {
    for (c = 0; c < cm->mi_cols << bsize[0]; ++c) {
#if CONFIG_AOM_HIGHBITDEPTH
      if (cm->use_highbitdepth) {
        src[r * stride + c] = CONVERT_TO_SHORTPTR(
            xd->plane[0].dst.buf)[r * xd->plane[0].dst.stride + c];
        ref_coeff[r * stride + c] =
            CONVERT_TO_SHORTPTR(ref->y_buffer)[r * ref->y_stride + c];
      } else {
#endif
        src[r * stride + c] =
            xd->plane[0].dst.buf[r * xd->plane[0].dst.stride + c];
        ref_coeff[r * stride + c] = ref->y_buffer[r * ref->y_stride + c];
#if CONFIG_AOM_HIGHBITDEPTH
      }
#endif
    }
  }
  nvsb = (cm->mi_rows + MAX_MIB_SIZE - 1) / MAX_MIB_SIZE;
  nhsb = (cm->mi_cols + MAX_MIB_SIZE - 1) / MAX_MIB_SIZE;
  /* Pick a base threshold based on the quantizer. The threshold will then be
     adjusted on a 64x64 basis. We use a threshold of the form T = a*Q^b,
     where a and b are derived empirically trying to optimize rate-distortion
     at different quantizer settings. */
  best_level = AOMMIN(
      MAX_DERING_LEVEL - 1,
      (int)floor(.5 +
                 .45 * pow(av1_ac_quant(cm->base_qindex, 0, cm->bit_depth) >>
                               (cm->bit_depth - 8),
                           0.6)));
  for (sbr = 0; sbr < nvsb; sbr++) {
    for (sbc = 0; sbc < nhsb; sbc++) {
      int nvb, nhb;
      int gi;
      int best_gi;
      int32_t best_mse = INT32_MAX;
      int16_t dst[MAX_MIB_SIZE * MAX_MIB_SIZE * 8 * 8];
      int16_t tmp_dst[MAX_MIB_SIZE * MAX_MIB_SIZE * 8 * 8];
      nhb = AOMMIN(MAX_MIB_SIZE, cm->mi_cols - MAX_MIB_SIZE * sbc);
      nvb = AOMMIN(MAX_MIB_SIZE, cm->mi_rows - MAX_MIB_SIZE * sbr);
      dering_count = sb_compute_dering_list(cm, sbr * MAX_MIB_SIZE,
                                            sbc * MAX_MIB_SIZE, dlist);
      if (dering_count == 0) continue;
      best_gi = 0;
      for (gi = 0; gi < DERING_REFINEMENT_LEVELS; gi++) {
        int cur_mse;
        int threshold;
        int16_t inbuf[OD_DERING_INBUF_SIZE];
        int16_t *in;
        int i, j;
        level = compute_level_from_index(best_level, gi);
        threshold = level << coeff_shift;
        for (r = 0; r < nvb << bsize[0]; r++) {
          for (c = 0; c < nhb << bsize[0]; c++) {
            dst[(r * MAX_MIB_SIZE << bsize[0]) + c] =
                src[((sbr * MAX_MIB_SIZE << bsize[0]) + r) * stride +
                    (sbc * MAX_MIB_SIZE << bsize[0]) + c];
          }
        }
        in = inbuf + OD_FILT_VBORDER * OD_FILT_BSTRIDE + OD_FILT_HBORDER;
        /* We avoid filtering the pixels for which some of the pixels to average
           are outside the frame. We could change the filter instead, but it
           would
           add special cases for any future vectorization. */
        for (i = 0; i < OD_DERING_INBUF_SIZE; i++)
          inbuf[i] = OD_DERING_VERY_LARGE;
        for (i = -OD_FILT_VBORDER * (sbr != 0);
             i < (nvb << bsize[0]) + OD_FILT_VBORDER * (sbr != nvsb - 1); i++) {
          for (j = -OD_FILT_HBORDER * (sbc != 0);
               j < (nhb << bsize[0]) + OD_FILT_HBORDER * (sbc != nhsb - 1);
               j++) {
            int16_t *x;
            x = &src[(sbr * stride * MAX_MIB_SIZE << bsize[0]) +
                     (sbc * MAX_MIB_SIZE << bsize[0])];
            in[i * OD_FILT_BSTRIDE + j] = x[i * stride + j];
          }
        }
        od_dering(tmp_dst, in, 0, dir, 0, dlist, dering_count, threshold,
                  coeff_shift);
        copy_dering_16bit_to_16bit(dst, MAX_MIB_SIZE << bsize[0], tmp_dst,
                                   dlist, dering_count, bsize[0]);
        compute_dist_wrap(
            dst, MAX_MIB_SIZE << bsize[0],
            &ref_coeff[(sbr * stride * MAX_MIB_SIZE << bsize[0]) +
                       (sbc * MAX_MIB_SIZE << bsize[0])],
            stride, nhb, nvb, coeff_shift);
        //printf("%d ", cur_mse);
        cur_mse = (int)compute_dist(
            dst, MAX_MIB_SIZE << bsize[0],
            &ref_coeff[(sbr * stride * MAX_MIB_SIZE << bsize[0]) +
                       (sbc * MAX_MIB_SIZE << bsize[0])],
            stride, nhb, nvb, coeff_shift);
        //printf("%d\n", cur_mse);
        if (cur_mse < best_mse) {
          best_gi = gi;
          best_mse = cur_mse;
        }
      }
      cm->mi_grid_visible[MAX_MIB_SIZE * sbr * cm->mi_stride +
                          MAX_MIB_SIZE * sbc]
          ->mbmi.dering_gain = best_gi;
    }
  }
  aom_free(src);
  aom_free(ref_coeff);
  return best_level;
}
