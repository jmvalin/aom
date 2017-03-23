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

#include "av1/common/clpf.h"
#include "./aom_dsp_rtcd.h"
#include "aom/aom_image.h"
#include "aom/aom_integer.h"
#include "av1/common/quant_common.h"

// Calculate the error of a filtered and unfiltered block
void aom_clpf_detect_c(const uint8_t *rec, const uint8_t *org, int rstride,
                       int ostride, int x0, int y0, int width, int height,
                       int *sum0, int *sum1, unsigned int strength, int size,
                       unsigned int dmp) {
  int x, y;
  for (y = y0; y < y0 + size; y++) {
    for (x = x0; x < x0 + size; x++) {
      const int O = org[y * ostride + x];
      const int X = rec[y * rstride + x];
      const int A = rec[AOMMAX(0, y - 2) * rstride + x];
      const int B = rec[AOMMAX(0, y - 1) * rstride + x];
      const int C = rec[y * rstride + AOMMAX(0, x - 2)];
      const int D = rec[y * rstride + AOMMAX(0, x - 1)];
      const int E = rec[y * rstride + AOMMIN(width - 1, x + 1)];
      const int F = rec[y * rstride + AOMMIN(width - 1, x + 2)];
      const int G = rec[AOMMIN(height - 1, y + 1) * rstride + x];
      const int H = rec[AOMMIN(height - 1, y + 2) * rstride + x];
      const int delta =
          av1_clpf_sample(X, A, B, C, D, E, F, G, H, strength, dmp);
      const int Y = X + delta;
      *sum0 += (O - X) * (O - X);
      *sum1 += (O - Y) * (O - Y);
    }
  }
}

void aom_clpf_detect_multi_c(const uint8_t *rec, const uint8_t *org,
                             int rstride, int ostride, int x0, int y0,
                             int width, int height, int *sum, int size,
                             unsigned int dmp) {
  int x, y;

  for (y = y0; y < y0 + size; y++) {
    for (x = x0; x < x0 + size; x++) {
      const int O = org[y * ostride + x];
      const int X = rec[y * rstride + x];
      const int A = rec[AOMMAX(0, y - 2) * rstride + x];
      const int B = rec[AOMMAX(0, y - 1) * rstride + x];
      const int C = rec[y * rstride + AOMMAX(0, x - 2)];
      const int D = rec[y * rstride + AOMMAX(0, x - 1)];
      const int E = rec[y * rstride + AOMMIN(width - 1, x + 1)];
      const int F = rec[y * rstride + AOMMIN(width - 1, x + 2)];
      const int G = rec[AOMMIN(height - 1, y + 1) * rstride + x];
      const int H = rec[AOMMIN(height - 1, y + 2) * rstride + x];
      const int delta1 = av1_clpf_sample(X, A, B, C, D, E, F, G, H, 1, dmp);
      const int delta2 = av1_clpf_sample(X, A, B, C, D, E, F, G, H, 2, dmp);
      const int delta3 = av1_clpf_sample(X, A, B, C, D, E, F, G, H, 4, dmp);
      const int F1 = X + delta1;
      const int F2 = X + delta2;
      const int F3 = X + delta3;
      sum[0] += (O - X) * (O - X);
      sum[1] += (O - F1) * (O - F1);
      sum[2] += (O - F2) * (O - F2);
      sum[3] += (O - F3) * (O - F3);
    }
  }
}

#if CONFIG_AOM_HIGHBITDEPTH
// Identical to aom_clpf_detect_c() apart from "rec" and "org".
void aom_clpf_detect_hbd_c(const uint16_t *rec, const uint16_t *org,
                           int rstride, int ostride, int x0, int y0, int width,
                           int height, int *sum0, int *sum1,
                           unsigned int strength, int size, unsigned int bd,
                           unsigned int dmp) {
  const int shift = bd - 8;
  int x, y;
  for (y = y0; y < y0 + size; y++) {
    for (x = x0; x < x0 + size; x++) {
      const int O = org[y * ostride + x] >> shift;
      const int X = rec[y * rstride + x] >> shift;
      const int A = rec[AOMMAX(0, y - 2) * rstride + x] >> shift;
      const int B = rec[AOMMAX(0, y - 1) * rstride + x] >> shift;
      const int C = rec[y * rstride + AOMMAX(0, x - 2)] >> shift;
      const int D = rec[y * rstride + AOMMAX(0, x - 1)] >> shift;
      const int E = rec[y * rstride + AOMMIN(width - 1, x + 1)] >> shift;
      const int F = rec[y * rstride + AOMMIN(width - 1, x + 2)] >> shift;
      const int G = rec[AOMMIN(height - 1, y + 1) * rstride + x] >> shift;
      const int H = rec[AOMMIN(height - 1, y + 2) * rstride + x] >> shift;
      const int delta = av1_clpf_sample(X, A, B, C, D, E, F, G, H,
                                        strength >> shift, dmp - shift);
      const int Y = X + delta;
      *sum0 += (O - X) * (O - X);
      *sum1 += (O - Y) * (O - Y);
    }
  }
}

// aom_clpf_detect_multi_c() apart from "rec" and "org".
void aom_clpf_detect_multi_hbd_c(const uint16_t *rec, const uint16_t *org,
                                 int rstride, int ostride, int x0, int y0,
                                 int width, int height, int *sum, int size,
                                 unsigned int bd, unsigned int dmp) {
  const int shift = bd - 8;
  int x, y;

  for (y = y0; y < y0 + size; y++) {
    for (x = x0; x < x0 + size; x++) {
      int O = org[y * ostride + x] >> shift;
      int X = rec[y * rstride + x] >> shift;
      const int A = rec[AOMMAX(0, y - 2) * rstride + x] >> shift;
      const int B = rec[AOMMAX(0, y - 1) * rstride + x] >> shift;
      const int C = rec[y * rstride + AOMMAX(0, x - 2)] >> shift;
      const int D = rec[y * rstride + AOMMAX(0, x - 1)] >> shift;
      const int E = rec[y * rstride + AOMMIN(width - 1, x + 1)] >> shift;
      const int F = rec[y * rstride + AOMMIN(width - 1, x + 2)] >> shift;
      const int G = rec[AOMMIN(height - 1, y + 1) * rstride + x] >> shift;
      const int H = rec[AOMMIN(height - 1, y + 2) * rstride + x] >> shift;
      const int delta1 =
          av1_clpf_sample(X, A, B, C, D, E, F, G, H, 1, dmp - shift);
      const int delta2 =
          av1_clpf_sample(X, A, B, C, D, E, F, G, H, 2, dmp - shift);
      const int delta3 =
          av1_clpf_sample(X, A, B, C, D, E, F, G, H, 4, dmp - shift);
      const int F1 = X + delta1;
      const int F2 = X + delta2;
      const int F3 = X + delta3;
      sum[0] += (O - X) * (O - X);
      sum[1] += (O - F1) * (O - F1);
      sum[2] += (O - F2) * (O - F2);
      sum[3] += (O - F3) * (O - F3);
    }
  }
}
#endif




