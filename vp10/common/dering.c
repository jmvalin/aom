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
  int xdec = 0;
  uint8_t bskip[4] = {0};
  int skip_stride = 0;
  int dir[OD_DERING_NBLOCKS][OD_DERING_NBLOCKS] = {{0}};
  int pli = 0;
  od_dering(&OD_DERING_VTBL_C, (int16_t *)y, ystride, (const int16_t *)x, xstride, ln, sbx, sby,
            nhsb, nvsb, xdec, dir, pli, &bskip[1], skip_stride,
            level, OD_DERING_NO_CHECK_OVERLAP);
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

void vp10_dering_sb2(YV12_BUFFER_CONFIG *frame_buffer,
                    VP10_COMMON *cm, MACROBLOCKD *xd,
                    MODE_INFO **mi_8x8, int mi_row, int mi_col, int level) {
  int r, c;
  int dir[OD_DERING_NBLOCKS][OD_DERING_NBLOCKS] = {{0}};
  unsigned char bskip[MI_BLOCK_SIZE][MI_BLOCK_SIZE];
  DECLARE_ALIGNED(16, int16_t, src[64 * 64]);
  DECLARE_ALIGNED(16, int16_t, dst[64 * 64]);
  for (r = 0; r < MI_BLOCK_SIZE; ++r) {
    for (c = 0; c < MI_BLOCK_SIZE; ++c) {
      const MB_MODE_INFO *mbmi =
          &mi_8x8[(mi_row + r) * cm->mi_stride + mi_col + c]->mbmi;
      bskip[r][c] = mbmi->skip;
    }
  }
  // Setup xd->plane pointers to the current mi
  vp10_setup_dst_planes(xd->plane, frame_buffer, mi_row, mi_col);
  for (r = 0; r < 64; ++r) {
    for (c = 0; c < 64; ++c) {
      src[r * 64 + c] = xd->plane[0].dst.buf[r * xd->plane[0].dst.stride + c];
    }
  }
  od_dering(&OD_DERING_VTBL_C, dst, 64, src, 64, 6, 0, 0, 1, 1, 0, dir, 0,
      &bskip[0][0], MI_BLOCK_SIZE, level, OD_DERING_NO_CHECK_OVERLAP);
  for (r = 0; r < 64; ++r) {
    for (c = 0; c < 64; ++c) {
      xd->plane[0].dst.buf[r * xd->plane[0].dst.stride + c] = dst[r * 64 + c];
    }
  }
}

void vp10_dering_rows(YV12_BUFFER_CONFIG *frame_buffer, VP10_COMMON *cm,
                      MACROBLOCKD *xd, int start, int stop, int level) {
  int mi_row, mi_col;
#if 0
  for (mi_row = start; mi_row < stop; mi_row += MI_BLOCK_SIZE) {
    for (mi_col = 0; mi_col < cm->mi_cols; mi_col += MI_BLOCK_SIZE) {
      vp10_dering_sb(frame_buffer, cm, xd, cm->mi_grid_visible, mi_row, mi_col,
                     level);
    }
  }
#else
  for (mi_row = start; mi_row + 7 < stop; mi_row += MI_BLOCK_SIZE) {
    for (mi_col = 0; mi_col + 7 < cm->mi_cols; mi_col += MI_BLOCK_SIZE) {
      vp10_dering_sb2(frame_buffer, cm, xd, cm->mi_grid_visible, mi_row, mi_col,
                     level);
    }
  }
#endif
}

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
