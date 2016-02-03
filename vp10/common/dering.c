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

void vp10_dering_frame(YV12_BUFFER_CONFIG *frame, VP10_COMMON *cm,
                       MACROBLOCKD *xd, int level) {
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
