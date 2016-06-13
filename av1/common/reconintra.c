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

#include "./aom_config.h"
#include "./aom_dsp_rtcd.h"

#if CONFIG_AOM_HIGHBITDEPTH
#include "aom_dsp/aom_dsp_common.h"
#endif  // CONFIG_AOM_HIGHBITDEPTH
#include "aom_mem/aom_mem.h"
#include "aom_ports/mem.h"
#include "aom_ports/aom_once.h"

#include "av1/common/reconintra.h"
#include "av1/common/onyxc_int.h"

enum {
  NEED_LEFT = 1 << 1,
  NEED_ABOVE = 1 << 2,
  NEED_ABOVERIGHT = 1 << 3,
  NEED_ABOVELEFT = 1 << 4,
  NEED_BOTTOMLEFT = 1 << 5,
};

static const uint8_t extend_modes[INTRA_MODES] = {
  NEED_ABOVE | NEED_LEFT,                   // DC
  NEED_ABOVE,                               // V
  NEED_LEFT,                                // H
  NEED_ABOVE | NEED_ABOVERIGHT,             // D45
  NEED_LEFT | NEED_ABOVE | NEED_ABOVELEFT,  // D135
  NEED_LEFT | NEED_ABOVE | NEED_ABOVELEFT,  // D117
  NEED_LEFT | NEED_ABOVE | NEED_ABOVELEFT,  // D153
  NEED_LEFT | NEED_BOTTOMLEFT,              // D207
  NEED_ABOVE | NEED_ABOVERIGHT,             // D63
  NEED_LEFT | NEED_ABOVE | NEED_ABOVELEFT,  // TM
};

static const uint8_t orders_64x64[1] = { 0 };
static const uint8_t orders_64x32[2] = { 0, 1 };
static const uint8_t orders_32x64[2] = { 0, 1 };
static const uint8_t orders_32x32[4] = {
  0, 1, 2, 3,
};
static const uint8_t orders_32x16[8] = {
  0, 2, 1, 3, 4, 6, 5, 7,
};
static const uint8_t orders_16x32[8] = {
  0, 1, 2, 3, 4, 5, 6, 7,
};
static const uint8_t orders_16x16[16] = {
  0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15,
};
static const uint8_t orders_16x8[32] = {
  0,  2,  8,  10, 1,  3,  9,  11, 4,  6,  12, 14, 5,  7,  13, 15,
  16, 18, 24, 26, 17, 19, 25, 27, 20, 22, 28, 30, 21, 23, 29, 31,
};
static const uint8_t orders_8x16[32] = {
  0,  1,  2,  3,  8,  9,  10, 11, 4,  5,  6,  7,  12, 13, 14, 15,
  16, 17, 18, 19, 24, 25, 26, 27, 20, 21, 22, 23, 28, 29, 30, 31,
};
static const uint8_t orders_8x8[64] = {
  0,  1,  4,  5,  16, 17, 20, 21, 2,  3,  6,  7,  18, 19, 22, 23,
  8,  9,  12, 13, 24, 25, 28, 29, 10, 11, 14, 15, 26, 27, 30, 31,
  32, 33, 36, 37, 48, 49, 52, 53, 34, 35, 38, 39, 50, 51, 54, 55,
  40, 41, 44, 45, 56, 57, 60, 61, 42, 43, 46, 47, 58, 59, 62, 63,
};
static const uint8_t *const orders[BLOCK_SIZES] = {
  orders_8x8,   orders_8x8,   orders_8x8,   orders_8x8,   orders_8x16,
  orders_16x8,  orders_16x16, orders_16x32, orders_32x16, orders_32x32,
  orders_32x64, orders_64x32, orders_64x64,
};

static int av1_has_right(BLOCK_SIZE bsize, int mi_row, int mi_col,
                         int right_available, TX_SIZE txsz, int y, int x,
                         int ss_x) {
  if (y == 0) {
    int wl = mi_width_log2_lookup[bsize];
    int hl = mi_height_log2_lookup[bsize];
    int w = 1 << (wl + 1 - ss_x);
    int step = 1 << txsz;
    const uint8_t *order = orders[bsize];
    int my_order, tr_order;

    if (x + step < w) return 1;

    mi_row = (mi_row & 7) >> hl;
    mi_col = (mi_col & 7) >> wl;

    if (mi_row == 0) return right_available;

    if (((mi_col + 1) << wl) >= 8) return 0;

    my_order = order[((mi_row + 0) << (3 - wl)) + mi_col + 0];
    tr_order = order[((mi_row - 1) << (3 - wl)) + mi_col + 1];

    return my_order > tr_order && right_available;
  } else {
    int wl = mi_width_log2_lookup[bsize];
    int w = 1 << (wl + 1 - ss_x);
    int step = 1 << txsz;

    return x + step < w;
  }
}

static int av1_has_bottom(BLOCK_SIZE bsize, int mi_row, int mi_col,
                          int bottom_available, TX_SIZE txsz, int y, int x,
                          int ss_y) {
  if (x == 0) {
    int wl = mi_width_log2_lookup[bsize];
    int hl = mi_height_log2_lookup[bsize];
    int h = 1 << (hl + 1 - ss_y);
    int step = 1 << txsz;
    const uint8_t *order = orders[bsize];
    int my_order, bl_order;

    mi_row = (mi_row & 7) >> hl;
    mi_col = (mi_col & 7) >> wl;

    if (mi_col == 0)
      return bottom_available &&
             (mi_row << (hl + !ss_y)) + y + step < (8 << !ss_y);

    if (((mi_row + 1) << hl) >= 8) return 0;

    if (y + step < h) return 1;

    my_order = order[((mi_row + 0) << (3 - wl)) + mi_col + 0];
    bl_order = order[((mi_row + 1) << (3 - wl)) + mi_col - 1];

    return bl_order < my_order && bottom_available;
  } else {
    return 0;
  }
}

typedef void (*intra_pred_fn)(uint8_t *dst, ptrdiff_t stride,
                              const uint8_t *above, const uint8_t *left);

static intra_pred_fn pred[INTRA_MODES][TX_SIZES];
static intra_pred_fn dc_pred[2][2][TX_SIZES];

#if CONFIG_AOM_HIGHBITDEPTH
typedef void (*intra_high_pred_fn)(uint16_t *dst, ptrdiff_t stride,
                                   const uint16_t *above, const uint16_t *left,
                                   int bd);
static intra_high_pred_fn pred_high[INTRA_MODES][4];
static intra_high_pred_fn dc_pred_high[2][2][4];
#endif  // CONFIG_AOM_HIGHBITDEPTH

static void av1_init_intra_predictors_internal(void) {
#define INIT_NO_4X4(p, type)                  \
  p[TX_8X8] = aom_##type##_predictor_8x8;     \
  p[TX_16X16] = aom_##type##_predictor_16x16; \
  p[TX_32X32] = aom_##type##_predictor_32x32

#define INIT_ALL_SIZES(p, type)           \
  p[TX_4X4] = aom_##type##_predictor_4x4; \
  INIT_NO_4X4(p, type)

  INIT_ALL_SIZES(pred[V_PRED], v);
  INIT_ALL_SIZES(pred[H_PRED], h);
#if CONFIG_MISC_FIXES
  INIT_ALL_SIZES(pred[D207_PRED], d207e);
  INIT_ALL_SIZES(pred[D45_PRED], d45e);
  INIT_ALL_SIZES(pred[D63_PRED], d63e);
#else
  INIT_ALL_SIZES(pred[D207_PRED], d207);
  INIT_ALL_SIZES(pred[D45_PRED], d45);
  INIT_ALL_SIZES(pred[D63_PRED], d63);
#endif
  INIT_ALL_SIZES(pred[D117_PRED], d117);
  INIT_ALL_SIZES(pred[D135_PRED], d135);
  INIT_ALL_SIZES(pred[D153_PRED], d153);
  INIT_ALL_SIZES(pred[TM_PRED], tm);

  INIT_ALL_SIZES(dc_pred[0][0], dc_128);
  INIT_ALL_SIZES(dc_pred[0][1], dc_top);
  INIT_ALL_SIZES(dc_pred[1][0], dc_left);
  INIT_ALL_SIZES(dc_pred[1][1], dc);

#if CONFIG_AOM_HIGHBITDEPTH
  INIT_ALL_SIZES(pred_high[V_PRED], highbd_v);
  INIT_ALL_SIZES(pred_high[H_PRED], highbd_h);
#if CONFIG_MISC_FIXES
  INIT_ALL_SIZES(pred_high[D207_PRED], highbd_d207e);
  INIT_ALL_SIZES(pred_high[D45_PRED], highbd_d45e);
  INIT_ALL_SIZES(pred_high[D63_PRED], highbd_d63);
#else
  INIT_ALL_SIZES(pred_high[D207_PRED], highbd_d207);
  INIT_ALL_SIZES(pred_high[D45_PRED], highbd_d45);
  INIT_ALL_SIZES(pred_high[D63_PRED], highbd_d63);
#endif
  INIT_ALL_SIZES(pred_high[D117_PRED], highbd_d117);
  INIT_ALL_SIZES(pred_high[D135_PRED], highbd_d135);
  INIT_ALL_SIZES(pred_high[D153_PRED], highbd_d153);
  INIT_ALL_SIZES(pred_high[TM_PRED], highbd_tm);

  INIT_ALL_SIZES(dc_pred_high[0][0], highbd_dc_128);
  INIT_ALL_SIZES(dc_pred_high[0][1], highbd_dc_top);
  INIT_ALL_SIZES(dc_pred_high[1][0], highbd_dc_left);
  INIT_ALL_SIZES(dc_pred_high[1][1], highbd_dc);
#endif  // CONFIG_AOM_HIGHBITDEPTH

#undef intra_pred_allsizes
}

#if CONFIG_AOM_HIGHBITDEPTH
static void build_intra_predictors_high(const MACROBLOCKD *xd,
                                        const uint8_t *ref8, int ref_stride,
                                        uint8_t *dst8, int dst_stride,
                                        PREDICTION_MODE mode, TX_SIZE tx_size,
                                        int n_top_px, int n_topright_px,
                                        int n_left_px, int n_bottomleft_px,
                                        int x, int y, int plane, int bd) {
  int i;
  uint16_t *dst = CONVERT_TO_SHORTPTR(dst8);
  uint16_t *ref = CONVERT_TO_SHORTPTR(ref8);
  DECLARE_ALIGNED(16, uint16_t, left_col[32]);
  DECLARE_ALIGNED(16, uint16_t, above_data[64 + 16]);
  uint16_t *above_row = above_data + 16;
  const uint16_t *const_above_row = above_row;
  const int bs = 4 << tx_size;
  const uint16_t *above_ref = ref - ref_stride;
  const int need_left = extend_modes[mode] & NEED_LEFT;
  const int need_above = extend_modes[mode] & NEED_ABOVE;
  const int need_aboveright = extend_modes[mode] & NEED_ABOVERIGHT;
  int base = 128 << (bd - 8);
  // 127 127 127 .. 127 127 127 127 127 127
  // 129  A   B  ..  Y   Z
  // 129  C   D  ..  W   X
  // 129  E   F  ..  U   V
  // 129  G   H  ..  S   T   T   T   T   T

  (void)x;
  (void)y;
  (void)plane;
  (void)need_left;
  (void)need_above;
  (void)need_aboveright;

  // NEED_LEFT
  if (extend_modes[mode] & NEED_LEFT) {
    const int need_bottom = !!(extend_modes[mode] & NEED_BOTTOMLEFT);
    i = 0;
    if (n_left_px > 0) {
      for (; i < n_left_px; i++) left_col[i] = ref[i * ref_stride - 1];
      if (need_bottom && n_bottomleft_px > 0) {
        assert(i == bs);
        for (; i < bs + n_bottomleft_px; i++)
          left_col[i] = ref[i * ref_stride - 1];
      }
      if (i < (bs << need_bottom))
        aom_memset16(&left_col[i], left_col[i - 1], (bs << need_bottom) - i);
    } else {
      aom_memset16(left_col, base + 1, bs << need_bottom);
    }
  }

  // NEED_ABOVE
  if (extend_modes[mode] & NEED_ABOVE) {
    const int need_right = !!(extend_modes[mode] & NEED_ABOVERIGHT);
    if (n_top_px > 0) {
      memcpy(above_row, above_ref, n_top_px * 2);
      i = n_top_px;
      if (need_right && n_topright_px > 0) {
        assert(n_top_px == bs);
        memcpy(above_row + bs, above_ref + bs, n_topright_px * 2);
        i += n_topright_px;
      }
      if (i < (bs << need_right))
        aom_memset16(&above_row[i], above_row[i - 1], (bs << need_right) - i);
    } else {
      aom_memset16(above_row, base - 1, bs << need_right);
    }
  }

  if (extend_modes[mode] & NEED_ABOVELEFT) {
    above_row[-1] =
        n_top_px > 0 ? (n_left_px > 0 ? above_ref[-1] : base + 1) : base - 1;
  }

  // predict
  if (mode == DC_PRED) {
    dc_pred_high[n_left_px > 0][n_top_px > 0][tx_size](
        dst, dst_stride, const_above_row, left_col, xd->bd);
  } else {
    pred_high[mode][tx_size](dst, dst_stride, const_above_row, left_col,
                             xd->bd);
  }
}
#endif  // CONFIG_AOM_HIGHBITDEPTH

static void build_intra_predictors(const MACROBLOCKD *xd, const uint8_t *ref,
                                   int ref_stride, uint8_t *dst, int dst_stride,
                                   PREDICTION_MODE mode, TX_SIZE tx_size,
                                   int n_top_px, int n_topright_px,
                                   int n_left_px, int n_bottomleft_px, int x,
                                   int y, int plane) {
  int i;
  DECLARE_ALIGNED(16, uint8_t, left_col[64]);
  const uint8_t *above_ref = ref - ref_stride;
  DECLARE_ALIGNED(16, uint8_t, above_data[64 + 16]);
  uint8_t *above_row = above_data + 16;
  const uint8_t *const_above_row = above_row;
  const int bs = 4 << tx_size;

  // 127 127 127 .. 127 127 127 127 127 127
  // 129  A   B  ..  Y   Z
  // 129  C   D  ..  W   X
  // 129  E   F  ..  U   V
  // 129  G   H  ..  S   T   T   T   T   T
  // ..

  (void)xd;
  (void)x;
  (void)y;
  (void)plane;
  assert(n_top_px >= 0);
  assert(n_topright_px >= 0);
  assert(n_left_px >= 0);
  assert(n_bottomleft_px >= 0);

  // NEED_LEFT
  if (extend_modes[mode] & NEED_LEFT) {
    const int need_bottom = !!(extend_modes[mode] & NEED_BOTTOMLEFT);
    i = 0;
    if (n_left_px > 0) {
      for (; i < n_left_px; i++) left_col[i] = ref[i * ref_stride - 1];
      if (need_bottom && n_bottomleft_px > 0) {
        assert(i == bs);
        for (; i < bs + n_bottomleft_px; i++)
          left_col[i] = ref[i * ref_stride - 1];
      }
      if (i < (bs << need_bottom))
        memset(&left_col[i], left_col[i - 1], (bs << need_bottom) - i);
    } else {
      memset(left_col, 129, bs << need_bottom);
    }
  }

  // NEED_ABOVE
  if (extend_modes[mode] & NEED_ABOVE) {
    const int need_right = !!(extend_modes[mode] & NEED_ABOVERIGHT);
    if (n_top_px > 0) {
      memcpy(above_row, above_ref, n_top_px);
      i = n_top_px;
      if (need_right && n_topright_px > 0) {
        assert(n_top_px == bs);
        memcpy(above_row + bs, above_ref + bs, n_topright_px);
        i += n_topright_px;
      }
      if (i < (bs << need_right))
        memset(&above_row[i], above_row[i - 1], (bs << need_right) - i);
    } else {
      memset(above_row, 127, bs << need_right);
    }
  }

  if (extend_modes[mode] & NEED_ABOVELEFT) {
    above_row[-1] = n_top_px > 0 ? (n_left_px > 0 ? above_ref[-1] : 129) : 127;
  }

  // predict
  if (mode == DC_PRED) {
    dc_pred[n_left_px > 0][n_top_px > 0][tx_size](dst, dst_stride,
                                                  const_above_row, left_col);
  } else {
    pred[mode][tx_size](dst, dst_stride, const_above_row, left_col);
  }
}

void av1_predict_intra_block(const MACROBLOCKD *xd, int bwl_in, int bhl_in,
                             TX_SIZE tx_size, PREDICTION_MODE mode,
                             const uint8_t *ref, int ref_stride, uint8_t *dst,
                             int dst_stride, int aoff, int loff, int plane) {
  const int txw = (1 << tx_size);
  const int have_top = loff || xd->up_available;
  const int have_left = aoff || xd->left_available;
  const int x = aoff * 4;
  const int y = loff * 4;
  const int bw = AOMMAX(2, 1 << bwl_in);
  const int bh = AOMMAX(2, 1 << bhl_in);
  const int mi_row = -xd->mb_to_top_edge >> 6;
  const int mi_col = -xd->mb_to_left_edge >> 6;
  const BLOCK_SIZE bsize = xd->mi[0]->mbmi.sb_type;
  const struct macroblockd_plane *const pd = &xd->plane[plane];
  const int right_available =
      mi_col + (bw >> !pd->subsampling_x) < xd->tile.mi_col_end;
  const int have_right = av1_has_right(bsize, mi_row, mi_col, right_available,
                                       tx_size, loff, aoff, pd->subsampling_x);
  const int have_bottom =
      av1_has_bottom(bsize, mi_row, mi_col, xd->mb_to_bottom_edge > 0, tx_size,
                     loff, aoff, pd->subsampling_y);
  const int wpx = 4 * bw;
  const int hpx = 4 * bh;
  const int txpx = 4 * txw;

  int xr = (xd->mb_to_right_edge >> (3 + pd->subsampling_x)) + (wpx - x - txpx);
  int yd =
      (xd->mb_to_bottom_edge >> (3 + pd->subsampling_y)) + (hpx - y - txpx);

#if CONFIG_AOM_HIGHBITDEPTH
  if (xd->cur_buf->flags & YV12_FLAG_HIGHBITDEPTH) {
    build_intra_predictors_high(xd, ref, ref_stride, dst, dst_stride, mode,
                                tx_size, have_top ? AOMMIN(txpx, xr + txpx) : 0,
                                have_top && have_right ? AOMMIN(txpx, xr) : 0,
                                have_left ? AOMMIN(txpx, yd + txpx) : 0,
                                have_bottom && have_left ? AOMMIN(txpx, yd) : 0,
                                x, y, plane, xd->bd);
    return;
  }
#endif
  build_intra_predictors(xd, ref, ref_stride, dst, dst_stride, mode, tx_size,
                         have_top ? AOMMIN(txpx, xr + txpx) : 0,
                         have_top && have_right ? AOMMIN(txpx, xr) : 0,
                         have_left ? AOMMIN(txpx, yd + txpx) : 0,
                         have_bottom && have_left ? AOMMIN(txpx, yd) : 0, x, y,
                         plane);
}

void av1_init_intra_predictors(void) {
  once(av1_init_intra_predictors_internal);
}