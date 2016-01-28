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

static void dering_helper_8x8(uint16_t *y, int ystride, const uint16_t *x,
                              int xstride, int has_top, int has_left,
                              int has_bottom, int has_right, int level) {
  int ln = 3;
  int sbx = has_left;
  int sby = has_top;
  int nhsb = has_left + has_right + 1;
  int nvsb = has_top + has_bottom + 1;
  int q = 1;
  int xdec = 0;
  uint8_t bskip[4] = {0};
  int skip_stride = 0;
  int dir[OD_DERING_NBLOCKS][OD_DERING_NBLOCKS] = {{0}};
  int pli = 0;
  od_dering(&OD_DERING_VTBL_C, (int16_t *)y, ystride, (const int16_t *)x, xstride, ln, sbx, sby,
            nhsb, nvsb, q, xdec, dir, pli, &bskip[1], skip_stride,
            (double)level, OD_DERING_NO_CHECK_OVERLAP);
}

static void dering_b_8x8(uint8_t *buf, int buf_stride, int has_top,
                         int has_left, int has_bottom, int has_right,
                         int level) {
  const int border = 3;
  DECLARE_ALIGNED(16, uint16_t, dering_buf[14 * 14]);
  DECLARE_ALIGNED(16, uint16_t, dst16_buf[14 * 14]);
  const int dering_stride = 8 + 2 * border;
  const int dst16_stride = 8 + 2 * border;
  uint16_t *dering = dering_buf + border * dering_stride + border;
  uint16_t *dst16 = dst16_buf + border * dst16_stride + border;
  int i, j;
  for (i = -border; i < 8 + border; ++i) {
    for (j = -border; j < 8 + border; ++j) {
      dst16[i * dst16_stride + j] = buf[i * buf_stride + j];
    }
  }
  dering_helper_8x8(dering, dering_stride, dst16, dst16_stride, has_top,
                    has_left, has_bottom, has_right, level);
  for (i = 0; i < 8; ++i) {
    for (j = 0; j < 8; ++j) {
      buf[i * buf_stride + j] = dering[i * dering_stride + j];
    }
  }
}

void vp10_dering_sb(YV12_BUFFER_CONFIG *frame_buffer,
                    VP10_COMMON *cm, MACROBLOCKD *xd,
                    MODE_INFO **mi_8x8, int mi_row, int mi_col, int level) {
  int r, c;
  for (r = 0; r < MI_BLOCK_SIZE && mi_row + r < cm->mi_rows; ++r) {
    for (c = 0; c < MI_BLOCK_SIZE && mi_col + c < cm->mi_cols; ++c) {
      int has_top = mi_row + r >= 1;
      int has_left = mi_col + c >= 1;
      int has_bottom = mi_row + r < cm->mi_rows - 1;
      int has_right = mi_col + c < cm-> mi_cols - 1;
      const MB_MODE_INFO *mbmi =
          &mi_8x8[(mi_row + r) * cm->mi_stride + mi_col + c]->mbmi;
      // Setup xd->plane pointers to the current mi
      vp10_setup_dst_planes(xd->plane, frame_buffer, mi_row + r, mi_col + c);
      // If this MI is prediction only don't filter
      if (mbmi->skip) continue;
      dering_b_8x8(xd->plane[0].dst.buf, xd->plane[0].dst.stride,
                       has_top, has_left, has_bottom, has_right, level);
    }
  }
}

void vp10_dering_rows(YV12_BUFFER_CONFIG *frame_buffer, VP10_COMMON *cm,
                      MACROBLOCKD *xd, int start, int stop, int level) {
  int mi_row, mi_col;
  for (mi_row = start; mi_row < stop; mi_row += MI_BLOCK_SIZE) {
    for (mi_col = 0; mi_col < cm->mi_cols; mi_col += MI_BLOCK_SIZE) {
      vp10_dering_sb(frame_buffer, cm, xd, cm->mi_grid_visible, mi_row, mi_col,
                     level);
    }
  }
}

void vp10_dering_frame(YV12_BUFFER_CONFIG *frame, VP10_COMMON *cm,
                       MACROBLOCKD *xd, int level) {
  vp10_dering_rows(frame, cm, xd, 0, cm->mi_rows, level);
}
