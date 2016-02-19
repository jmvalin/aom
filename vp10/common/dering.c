/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <string.h>

#include "./vpx_scale_rtcd.h"
#include "vpx/vpx_integer.h"
#include "vp10/common/dering.h"
#include "vp10/common/onyxc_int.h"
#include "vp10/common/reconinter.h"
#include "od_dering.h"

double dering_gains[4] = {0, .6, 1, 1.7};

static double compute_dist(int16_t *x, int16_t *y,
 int n) {
  int i;
  double sum;
  sum = 0;
  for (i = 0; i < n*n; i++) {
    double tmp;
    tmp = x[i] - y[i];
    sum += tmp*tmp;
  }
  return sum;
}

int vp10_dering_search(YV12_BUFFER_CONFIG *frame, const YV12_BUFFER_CONFIG *ref,
                       VP10_COMMON *cm,
                       MACROBLOCKD *xd, int *dering_level) {
  int r, c;
  int sbr, sbc;
  int nhsb, nvsb;
  int16_t *src, *dst, *ref_coeff;
  unsigned char *bskip;
  int dir[OD_DERING_NBLOCKS][OD_DERING_NBLOCKS] = {{0}};
  int stride;
  int (*mse)[MAX_DERING_LEVEL];
  int best_count[MAX_DERING_LEVEL] = {0};
  int level;
  int best_level;
  int level_cdf;
  src = malloc(sizeof(*src)*cm->mi_rows*cm->mi_cols*64);
  dst = malloc(sizeof(*dst)*cm->mi_rows*cm->mi_cols*64);
  ref_coeff = malloc(sizeof(*dst)*cm->mi_rows*cm->mi_cols*64);
  bskip = malloc(sizeof(*bskip)*cm->mi_rows*cm->mi_cols);
  vp10_setup_dst_planes(xd->plane, frame, 0, 0);
  stride = 8*cm->mi_cols;
  for (r = 0; r < 8*cm->mi_rows; ++r) {
    for (c = 0; c < 8*cm->mi_cols; ++c) {
      dst[r * stride + c] = src[r * stride + c] = xd->plane[0].dst.buf[r * xd->plane[0].dst.stride + c];
      ref_coeff[r * stride + c] = ref->y_buffer[r * ref->y_stride + c];
    }
  }
  for (r = 0; r < cm->mi_rows; ++r) {
    for (c = 0; c < cm->mi_cols; ++c) {
      const MB_MODE_INFO *mbmi =
          &cm->mi_grid_visible[r * cm->mi_stride + c]->mbmi;
      bskip[r * cm->mi_cols + c] = mbmi->skip;
    }
  }
  nvsb = cm->mi_rows/MI_BLOCK_SIZE;
  nhsb = cm->mi_cols/MI_BLOCK_SIZE;
  mse = malloc(nvsb*nhsb*sizeof(*mse));
  for (sbr = 0; sbr < nvsb; sbr++) {
    for (sbc = 0; sbc < nhsb; sbc++) {
      int best_mse = 1000000000;
      best_level = 0;
      for (level = 0; level < 64; level++) {
        od_dering(&OD_DERING_VTBL_C, dst + sbr*stride*8*MI_BLOCK_SIZE + sbc*8*MI_BLOCK_SIZE,
            cm->mi_cols*8, src + sbr*stride*8*MI_BLOCK_SIZE + sbc*8*MI_BLOCK_SIZE, cm->mi_cols*8, 6,
            sbc, sbr, nhsb, nvsb, 0, dir, 0,
            bskip + MI_BLOCK_SIZE*sbr*cm->mi_cols + MI_BLOCK_SIZE*sbc, cm->mi_cols, level, OD_DERING_NO_CHECK_OVERLAP);
        mse[nhsb*sbr+sbc][level] = compute_dist(dst + sbr*stride*8*MI_BLOCK_SIZE + sbc*8*MI_BLOCK_SIZE,
                           ref_coeff + sbr*stride*8*MI_BLOCK_SIZE + sbc*8*MI_BLOCK_SIZE,
                           8*MI_BLOCK_SIZE);
        if (mse[nhsb*sbr+sbc][level] < best_mse) {
          best_mse = mse[nhsb*sbr+sbc][level];
          best_level = level;
        }
      }
      best_count[best_level]++;
    }
  }
  level_cdf = 0;
  //Find the median of the best level
  for (level = 0; level < MAX_DERING_LEVEL; level++) {
    level_cdf += best_count[level];
    if (level_cdf > nvsb*nhsb/2)
      break;
  }
  best_level = level*1.3;
  for (sbr = 0; sbr < nvsb; sbr++) {
    for (sbc = 0; sbc < nhsb; sbc++) {
      int gi;
      int best_gi;
      int best_mse = mse[nhsb*sbr+sbc][0];
      best_gi = 0;
      for (gi = 1; gi < 4; gi++) {
        level = (int)(.5 + best_level * dering_gains[gi]);
        if (mse[nhsb*sbr+sbc][level] < best_mse) {
          best_gi = gi;
          best_mse = mse[nhsb*sbr+sbc][level];
        }
      }
      dering_level[nhsb*sbr+sbc] = best_gi;
    }
  }
  free(src);
  free(dst);
  free(ref_coeff);
  free(bskip);
  return best_level;
}

void vp10_dering_frame(YV12_BUFFER_CONFIG *frame, VP10_COMMON *cm,
                       MACROBLOCKD *xd, int global_level, int *dering_level) {
  int r, c;
  int sbr, sbc;
  int nhsb, nvsb;
  int16_t *src, *dst;
  unsigned char *bskip;
  int dir[OD_DERING_NBLOCKS][OD_DERING_NBLOCKS] = {{0}};
  int stride;
  src = malloc(sizeof(*src)*cm->mi_rows*cm->mi_cols*64);
  dst = malloc(sizeof(*dst)*cm->mi_rows*cm->mi_cols*64);
  bskip = malloc(sizeof(*bskip)*cm->mi_rows*cm->mi_cols);
  vp10_setup_dst_planes(xd->plane, frame, 0, 0);
  stride = 8*cm->mi_cols;
  for (r = 0; r < 8*cm->mi_rows; ++r) {
    for (c = 0; c < 8*cm->mi_cols; ++c) {
      dst[r * stride + c] = src[r * stride + c] = xd->plane[0].dst.buf[r * xd->plane[0].dst.stride + c];
    }
  }
  for (r = 0; r < cm->mi_rows; ++r) {
    for (c = 0; c < cm->mi_cols; ++c) {
      const MB_MODE_INFO *mbmi =
          &cm->mi_grid_visible[r * cm->mi_stride + c]->mbmi;
      bskip[r * cm->mi_cols + c] = mbmi->skip;
    }
  }
  nvsb = cm->mi_rows/MI_BLOCK_SIZE;
  nhsb = cm->mi_cols/MI_BLOCK_SIZE;
  for (sbr = 0; sbr < nvsb; sbr++) {
    for (sbc = 0; sbc < nhsb; sbc++) {
      int level;
      if (dering_level) {
        level = (int)(.5 + global_level * dering_gains[dering_level[nhsb*sbr+sbc]]);
      }
      else level = global_level;
      od_dering(&OD_DERING_VTBL_C, dst + sbr*stride*8*MI_BLOCK_SIZE + sbc*8*MI_BLOCK_SIZE,
          cm->mi_cols*8, src + sbr*stride*8*MI_BLOCK_SIZE + sbc*8*MI_BLOCK_SIZE, cm->mi_cols*8, 6,
          sbc, sbr, nhsb, nvsb, 0, dir, 0,
          bskip + MI_BLOCK_SIZE*sbr*cm->mi_cols + MI_BLOCK_SIZE*sbc, cm->mi_cols, level, OD_DERING_NO_CHECK_OVERLAP);
    }
  }
  for (r = 0; r < 8*cm->mi_rows; ++r) {
    for (c = 0; c < 8*cm->mi_cols; ++c) {
      xd->plane[0].dst.buf[r * xd->plane[0].dst.stride + c] = dst[r * stride + c];
    }
  }
  free(src);
  free(dst);
  free(bskip);
}
