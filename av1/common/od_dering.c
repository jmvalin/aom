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
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include <math.h>
#include "dering.h"

const od_dering_opt_vtbl OD_DERING_VTBL_C = {
  { od_filter_dering_direction_4x4_c, od_filter_dering_direction_8x8_c },
  { od_filter_dering_orthogonal_4x4_c, od_filter_dering_orthogonal_8x8_c }
};

/* Generated from gen_filter_tables.c. */
const int OD_DIRECTION_OFFSETS_TABLE[8][3] = {
  { -1 * OD_FILT_BSTRIDE + 1, -2 * OD_FILT_BSTRIDE + 2,
    -3 * OD_FILT_BSTRIDE + 3 },
  { 0 * OD_FILT_BSTRIDE + 1, -1 * OD_FILT_BSTRIDE + 2,
    -1 * OD_FILT_BSTRIDE + 3 },
  { 0 * OD_FILT_BSTRIDE + 1, 0 * OD_FILT_BSTRIDE + 2, 0 * OD_FILT_BSTRIDE + 3 },
  { 0 * OD_FILT_BSTRIDE + 1, 1 * OD_FILT_BSTRIDE + 2, 1 * OD_FILT_BSTRIDE + 3 },
  { 1 * OD_FILT_BSTRIDE + 1, 2 * OD_FILT_BSTRIDE + 2, 3 * OD_FILT_BSTRIDE + 3 },
  { 1 * OD_FILT_BSTRIDE + 0, 2 * OD_FILT_BSTRIDE + 1, 3 * OD_FILT_BSTRIDE + 1 },
  { 1 * OD_FILT_BSTRIDE + 0, 2 * OD_FILT_BSTRIDE + 0, 3 * OD_FILT_BSTRIDE + 0 },
  { 1 * OD_FILT_BSTRIDE + 0, 2 * OD_FILT_BSTRIDE - 1, 3 * OD_FILT_BSTRIDE - 1 },
};

const double OD_DERING_GAIN_TABLE[OD_DERING_LEVELS] = { 0, 0.5,  0.707,
                                                        1, 1.41, 2 };

#define ENABLE_SSE2

#ifdef ENABLE_SSE2
#include <smmintrin.h>
#include <emmintrin.h>
#include <tmmintrin.h>
#include "aom_dsp/x86/inv_txfm_sse2.h"

static INLINE __m128i fold_mul_and_sum(__m128i partiala, __m128i partialb,
    __m128i const1, __m128i const2) {
  __m128i tmp;
  partialb = _mm_shuffle_epi8(partialb, _mm_set_epi8(15, 14, 1, 0, 3, 2, 5, 4,
      7, 6, 9, 8, 11, 10, 13, 12));
  tmp = partiala;
  partiala = _mm_unpacklo_epi16(partiala, partialb);
  partialb = _mm_unpackhi_epi16(tmp, partialb);
  partiala = _mm_madd_epi16(partiala, partiala);
  partialb = _mm_madd_epi16(partialb, partialb);
  partiala = _mm_mullo_epi32(partiala, const1);
  partialb = _mm_mullo_epi32(partialb, const2);
  partiala = _mm_add_epi32(partiala, partialb);
  partiala = _mm_add_epi32(partiala,
      _mm_unpackhi_epi64(partiala, partiala));
  partiala = _mm_add_epi32(partiala,
      _mm_shufflelo_epi16(partiala, _MM_SHUFFLE(1, 0, 3, 2)));
  return partiala;
}

static INLINE void compute_directions(__m128i lines[8], int32_t tmp_cost1[3]) {
  __m128i partial0a, partial0b, partial5a, partial5b, partial7a, partial7b;
  __m128i tmp;
  partial0a = lines[0];
  partial0a = _mm_add_epi16(partial0a, _mm_slli_si128(lines[1], 2));
  partial0b = _mm_srli_si128(lines[1], 14);
  tmp = _mm_add_epi16(lines[0], lines[1]);
  partial5a = _mm_slli_si128(tmp, 10);
  partial5b = _mm_srli_si128(tmp, 6);
  partial7a = _mm_slli_si128(tmp, 4);
  partial7b = _mm_srli_si128(tmp, 12);

  partial0a = _mm_add_epi16(partial0a, _mm_slli_si128(lines[2], 4));
  partial0b = _mm_add_epi16(partial0b, _mm_srli_si128(lines[2], 12));
  partial0a = _mm_add_epi16(partial0a, _mm_slli_si128(lines[3], 6));
  partial0b = _mm_add_epi16(partial0b, _mm_srli_si128(lines[3], 10));
  tmp = _mm_add_epi16(lines[2], lines[3]);
  partial5a = _mm_add_epi16(partial5a, _mm_slli_si128(tmp, 8));
  partial5b = _mm_add_epi16(partial5b, _mm_srli_si128(tmp, 8));
  partial7a = _mm_add_epi16(partial7a, _mm_slli_si128(tmp, 6));
  partial7b = _mm_add_epi16(partial7b, _mm_srli_si128(tmp, 10));

  partial0a = _mm_add_epi16(partial0a, _mm_slli_si128(lines[4], 8));
  partial0b = _mm_add_epi16(partial0b, _mm_srli_si128(lines[4], 8));
  partial0a = _mm_add_epi16(partial0a, _mm_slli_si128(lines[5], 10));
  partial0b = _mm_add_epi16(partial0b, _mm_srli_si128(lines[5], 6));
  tmp = _mm_add_epi16(lines[4], lines[5]);
  partial5a = _mm_add_epi16(partial5a, _mm_slli_si128(tmp, 6));
  partial5b = _mm_add_epi16(partial5b, _mm_srli_si128(tmp, 10));
  partial7a = _mm_add_epi16(partial7a, _mm_slli_si128(tmp, 8));
  partial7b = _mm_add_epi16(partial7b, _mm_srli_si128(tmp, 8));

  partial0a = _mm_add_epi16(partial0a, _mm_slli_si128(lines[6], 12));
  partial0b = _mm_add_epi16(partial0b, _mm_srli_si128(lines[6], 4));
  partial0a = _mm_add_epi16(partial0a, _mm_slli_si128(lines[7], 14));
  partial0b = _mm_add_epi16(partial0b, _mm_srli_si128(lines[7], 2));
  tmp = _mm_add_epi16(lines[6], lines[7]);
  partial5a = _mm_add_epi16(partial5a, _mm_slli_si128(tmp, 4));
  partial5b = _mm_add_epi16(partial5b, _mm_srli_si128(tmp, 12));
  partial7a = _mm_add_epi16(partial7a, _mm_slli_si128(tmp, 10));
  partial7b = _mm_add_epi16(partial7b, _mm_srli_si128(tmp, 6));
  partial0a = fold_mul_and_sum(partial0a, partial0b, _mm_set_epi32(210, 280, 420, 840), _mm_set_epi32(105, 120, 140, 168));
  partial7a = fold_mul_and_sum(partial7a, partial7b, _mm_set_epi32(210, 420, 0, 0), _mm_set_epi32(105, 105, 105, 140));
  partial5a = fold_mul_and_sum(partial5a, partial5b, _mm_set_epi32(210, 420, 0, 0), _mm_set_epi32(105, 105, 105, 140));
  tmp_cost1[0] = _mm_cvtsi128_si32(partial0a);
  tmp_cost1[1] = _mm_cvtsi128_si32(partial5a);
  tmp_cost1[2] = _mm_cvtsi128_si32(partial7a);
}

int od_dir_find8_sse2(const od_dering_in *img, int stride, int32_t *var,
                        int coeff_shift) {
  int i;
  int32_t cost[8] = { 0 };
  int partial[8][15] = { { 0 } };
  int32_t best_cost = 0;
  int32_t tmp_cost1[3];
  int32_t tmp_cost2[3];
  int best_dir = 0;
  __m128i lines[8], tlines[8];
  __m128i tmp;
  __m128i partial2;
  __m128i partial6;
  __m128i partial4a, partial4b;
  /* Instead of dividing by n between 2 and 8, we multiply by 3*5*7*8/n.
     The output is then 840 times larger, but we don't care for finding
     the max. */
  static const int div_table[] = { 0, 840, 420, 280, 210, 168, 140, 120, 105 };

  partial6 = _mm_setzero_si128();
  for (i = 0; i < 8; i++) {
    lines[i] = _mm_loadu_si128((__m128i*)&img[i * stride]);
    /* Shift here. */
    lines[i] = _mm_sub_epi16(lines[i], _mm_set1_epi16(128));
    partial6 = _mm_add_epi16(partial6, lines[i]);
  }
  partial6 = _mm_madd_epi16(partial6, partial6);
  partial6 = _mm_add_epi32(partial6, _mm_unpackhi_epi64(partial6, partial6));
  partial6 = _mm_add_epi32(partial6, _mm_shufflelo_epi16(partial6, _MM_SHUFFLE(1, 0, 3, 2)));
  cost[6] = _mm_cvtsi128_si32(partial6);

  partial4a = lines[0];
  partial4a = _mm_add_epi16(partial4a, _mm_srli_si128(lines[1], 2));
  partial4b = _mm_slli_si128(lines[1], 14);
  partial4a = _mm_add_epi16(partial4a, _mm_srli_si128(lines[2], 4));
  partial4b = _mm_add_epi16(partial4b, _mm_slli_si128(lines[2], 12));
  partial4a = _mm_add_epi16(partial4a, _mm_srli_si128(lines[3], 6));
  partial4b = _mm_add_epi16(partial4b, _mm_slli_si128(lines[3], 10));
  partial4a = _mm_add_epi16(partial4a, _mm_srli_si128(lines[4], 8));
  partial4b = _mm_add_epi16(partial4b, _mm_slli_si128(lines[4], 8));
  partial4a = _mm_add_epi16(partial4a, _mm_srli_si128(lines[5], 10));
  partial4b = _mm_add_epi16(partial4b, _mm_slli_si128(lines[5], 6));
  partial4a = _mm_add_epi16(partial4a, _mm_srli_si128(lines[6], 12));
  partial4b = _mm_add_epi16(partial4b, _mm_slli_si128(lines[6], 4));
  partial4a = _mm_add_epi16(partial4a, _mm_srli_si128(lines[7], 14));
  partial4b = _mm_add_epi16(partial4b, _mm_slli_si128(lines[7], 2));
  /* Shift by one position and reverse vector. */
  partial4b = _mm_shuffle_epi8(partial4b, _mm_set_epi8(3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 1, 0));
  tmp = partial4a;
  partial4a = _mm_unpacklo_epi16(partial4a, partial4b);
  partial4b = _mm_unpackhi_epi16(tmp, partial4b);
  if (0) {
    int16_t tmp[16];
    _mm_storeu_si128((__m128i*)tmp, partial4a);
    _mm_storeu_si128((__m128i*)&tmp[8], partial4b);
    for (i=0;i<16;i++) {
      printf("%d ", tmp[i]);
    }
    printf("\n");
  }
  partial4a = _mm_madd_epi16(partial4a, partial4a);
  partial4b = _mm_madd_epi16(partial4b, partial4b);
  partial4a = _mm_mullo_epi32(partial4a, _mm_setr_epi32(105, 120, 140, 168));
  partial4b = _mm_mullo_epi32(partial4b, _mm_setr_epi32(210, 280, 420, 840));
  partial4a = _mm_add_epi32(partial4a, partial4b);
  partial4a = _mm_add_epi32(partial4a, _mm_unpackhi_epi64(partial4a, partial4a));
  partial4a = _mm_add_epi32(partial4a, _mm_shufflelo_epi16(partial4a, _MM_SHUFFLE(1, 0, 3, 2)));
  cost[4] = _mm_cvtsi128_si32(partial4a);
  //printf("%d\n", _mm_cvtsi128_si32(partial4a));

  compute_directions(lines, tmp_cost1);

  array_transpose_8x8(lines, tlines);
  for (i = 0; i < 8; i++) {
    partial2 = _mm_add_epi16(partial2, tlines[i]);
  }
  partial2 = _mm_madd_epi16(partial2, partial2);
  partial2 = _mm_add_epi32(partial2, _mm_unpackhi_epi64(partial2, partial2));
  partial2 = _mm_add_epi32(partial2, _mm_shufflelo_epi16(partial2, _MM_SHUFFLE(1, 0, 3, 2)));
  cost[2] = _mm_cvtsi128_si32(partial2);

  compute_directions(tlines, tmp_cost2);

  //printf("%d\n", _mm_cvtsi128_si32(partial2));
#if 0
  for (i = 0; i < 8; i++) {
    int j;
    for (j = 0; j < 8; j++) {
      int x;
      /* We subtract 128 here to reduce the maximum range of the squared
         partial sums. */
      x = (img[i * stride + j] >> coeff_shift) - 128;
      //partial[0][i + j] += x;
      partial[1][i + j / 2] += x;
      //partial[2][i] += x;
      partial[3][3 + i - j / 2] += x;
      //partial[4][7 + i - j] += x;
      //partial[5][3 - i / 2 + j] += x;
      //partial[6][j] += x;
      //partial[7][i / 2 + j] += x;
    }
  }
#endif
#if 0
  for (i=0;i<15;i++) {
    printf("%d ", partial[5][i]);
  }
  printf("\n\n");
#endif
  for (i = 0; i < 8; i++) {
    //cost[2] += partial[2][i] * partial[2][i];
    //cost[6] += partial[6][i] * partial[6][i];
  }
  //printf("%d\n\n", cost[2]);
  cost[2] *= div_table[8];
  cost[6] *= div_table[8];
  for (i = 0; i < 7; i++) {
    //cost[0] += (partial[0][i] * partial[0][i] +
    //            partial[0][14 - i] * partial[0][14 - i]) *
    //           div_table[i + 1];
    //cost[4] += (partial[4][i] * partial[4][i] +
    //            partial[4][14 - i] * partial[4][14 - i]) *
    //           div_table[i + 1];
  }
  //cost[0] += partial[0][7] * partial[0][7] * div_table[8];
  //cost[4] += partial[4][7] * partial[4][7] * div_table[8];
  /*for (i = 1; i < 8; i += 2) {
    int j;
    for (j = 0; j < 4 + 1; j++) {
      cost[i] += partial[i][3 + j] * partial[i][3 + j];
    }
    cost[i] *= div_table[8];
    for (j = 0; j < 4 - 1; j++) {
      cost[i] += (partial[i][j] * partial[i][j] +
                  partial[i][10 - j] * partial[i][10 - j]) *
                 div_table[2 * j + 2];
    }
  }*/
  //printf("%d\n\n", cost[7]);
  cost[0] = tmp_cost1[0];
  cost[5] = tmp_cost1[1];
  cost[7] = tmp_cost1[2];
  cost[1] = tmp_cost2[2];
  cost[3] = tmp_cost2[1];
  for (i = 0; i < 8; i++) {
    if (cost[i] > best_cost) {
      best_cost = cost[i];
      best_dir = i;
    }
  }
  /* Difference between the optimal variance and the variance along the
     orthogonal direction. Again, the sum(x^2) terms cancel out. */
  *var = best_cost - cost[(best_dir + 4) & 7];
  /* We'd normally divide by 840, but dividing by 1024 is close enough
     for what we're going to do with this. */
  *var >>= 10;
  return best_dir;
}
#endif

/* Detect direction. 0 means 45-degree up-right, 2 is horizontal, and so on.
   The search minimizes the weighted variance along all the lines in a
   particular direction, i.e. the squared error between the input and a
   "predicted" block where each pixel is replaced by the average along a line
   in a particular direction. Since each direction have the same sum(x^2) term,
   that term is never computed. See Section 2, step 2, of:
   http://jmvalin.ca/notes/intra_paint.pdf */
int od_dir_find8(const od_dering_in *img, int stride, int32_t *var,
                        int coeff_shift) {
  int i;
  int32_t cost[8] = { 0 };
  int partial[8][15] = { { 0 } };
  int32_t best_cost = 0;
  int best_dir = 0;
  /* Instead of dividing by n between 2 and 8, we multiply by 3*5*7*8/n.
     The output is then 840 times larger, but we don't care for finding
     the max. */
  static const int div_table[] = { 0, 840, 420, 280, 210, 168, 140, 120, 105 };
  for (i = 0; i < 8; i++) {
    int j;
    for (j = 0; j < 8; j++) {
      int x;
      /* We subtract 128 here to reduce the maximum range of the squared
         partial sums. */
      x = (img[i * stride + j] >> coeff_shift) - 128;
      partial[0][i + j] += x;
      partial[1][i + j / 2] += x;
      partial[2][i] += x;
      partial[3][3 + i - j / 2] += x;
      partial[4][7 + i - j] += x;
      partial[5][3 - i / 2 + j] += x;
      partial[6][j] += x;
      partial[7][i / 2 + j] += x;
    }
  }
  for (i = 0; i < 8; i++) {
    cost[2] += partial[2][i] * partial[2][i];
    cost[6] += partial[6][i] * partial[6][i];
  }
  cost[2] *= div_table[8];
  cost[6] *= div_table[8];
  for (i = 0; i < 7; i++) {
    cost[0] += (partial[0][i] * partial[0][i] +
                partial[0][14 - i] * partial[0][14 - i]) *
               div_table[i + 1];
    cost[4] += (partial[4][i] * partial[4][i] +
                partial[4][14 - i] * partial[4][14 - i]) *
               div_table[i + 1];
  }
  cost[0] += partial[0][7] * partial[0][7] * div_table[8];
  cost[4] += partial[4][7] * partial[4][7] * div_table[8];
  for (i = 1; i < 8; i += 2) {
    int j;
    for (j = 0; j < 4 + 1; j++) {
      cost[i] += partial[i][3 + j] * partial[i][3 + j];
    }
    cost[i] *= div_table[8];
    for (j = 0; j < 4 - 1; j++) {
      cost[i] += (partial[i][j] * partial[i][j] +
                  partial[i][10 - j] * partial[i][10 - j]) *
                 div_table[2 * j + 2];
    }
  }
  for (i = 0; i < 8; i++) {
    if (cost[i] > best_cost) {
      best_cost = cost[i];
      best_dir = i;
    }
  }
  /* Difference between the optimal variance and the variance along the
     orthogonal direction. Again, the sum(x^2) terms cancel out. */
  *var = best_cost - cost[(best_dir + 4) & 7];
  /* We'd normally divide by 840, but dividing by 1024 is close enough
     for what we're going to do with this. */
  *var >>= 10;
  return best_dir;
}

#define OD_DERING_VERY_LARGE (30000)
#define OD_DERING_INBUF_SIZE \
  ((OD_BSIZE_MAX + 2 * OD_FILT_BORDER) * (OD_BSIZE_MAX + 2 * OD_FILT_BORDER))

#ifdef ENABLE_SSE2

# if OD_GNUC_PREREQ(3, 0, 0)
#  define OD_SIMD_INLINE static __inline __attribute__((always_inline))
# else
#  define OD_SIMD_INLINE static
# endif

/*Corresponds to _mm_abs_epi16 (ssse3).*/
OD_SIMD_INLINE __m128i od_abs_epi16(__m128i in) {
  __m128i mask;
  mask = _mm_cmpgt_epi16(_mm_setzero_si128(), in);
  return _mm_sub_epi16(_mm_xor_si128(in, mask), mask);
}

OD_SIMD_INLINE __m128i od_cmplt_abs_epi16(__m128i in, __m128i threshold) {
  return _mm_and_si128(_mm_cmplt_epi16(_mm_sub_epi16(_mm_setzero_si128(),
   threshold), in), _mm_cmplt_epi16(in, threshold));
}

void od_filter_dering_direction_8x8_sse2(int16_t *y, int ystride,
 const int16_t *in, int threshold, int dir) {
  int i;
  int k;
  static const int taps[3] = {3, 2, 1};
  __m128i sum;
  __m128i p;
  __m128i cmp;
  __m128i row;
  __m128i res;
  __m128i thresh;
  thresh = _mm_set1_epi16(threshold);
  for (i = 0; i < 8; i++) {
    sum = _mm_set1_epi16(0);
    row = _mm_loadu_si128((__m128i*)&in[i*OD_FILT_BSTRIDE]);
    for (k = 0; k < 3; k++) {
      /*p = in[i*OD_FILT_BSTRIDE + offset] - row*/;
      p = _mm_sub_epi16(_mm_loadu_si128((__m128i*)&in[i*OD_FILT_BSTRIDE +
       OD_DIRECTION_OFFSETS_TABLE[dir][k]]), row);
      /*if (abs(p) < thresh) sum += taps[k]*p*/
      cmp = od_cmplt_abs_epi16(p, thresh);
      p = _mm_mullo_epi16(p, _mm_set1_epi16(taps[k]));
      p = _mm_and_si128(p, cmp);
      sum = _mm_add_epi16(sum, p);
      /*p = in[i*OD_FILT_BSTRIDE - offset] - row*/;
      p = _mm_sub_epi16(_mm_loadu_si128((__m128i*)&in[i*OD_FILT_BSTRIDE -
       OD_DIRECTION_OFFSETS_TABLE[dir][k]]), row);
      /*if (abs(p) < thresh) sum += taps[k]*p1*/
      cmp = od_cmplt_abs_epi16(p, thresh);
      p = _mm_mullo_epi16(p, _mm_set1_epi16(taps[k]));
      p = _mm_and_si128(p, cmp);
      sum = _mm_add_epi16(sum, p);
    }
    /*res = row + ((sum + 8) >> 4)*/
    res = _mm_add_epi16(sum, _mm_set1_epi16(8));
    res = _mm_srai_epi16(res, 4);
    res = _mm_add_epi16(row, res);
    _mm_storeu_si128((__m128i*)&y[i*ystride], res);
  }
}

void od_filter_dering_direction_4x4_sse2(int16_t *y, int ystride,
 const int16_t *in, int threshold, int dir) {
  int i;
  int k;
  static const int taps[3] = {3, 2, 1};
  __m128i sum;
  __m128i p;
  __m128i cmp;
  __m128i row;
  __m128i res;
  __m128i thresh;
  thresh = _mm_set1_epi16(threshold);
  for (i = 0; i < 4; i++) {
    sum = _mm_set1_epi16(0);
    row = _mm_loadl_epi64((__m128i*)&in[i*OD_FILT_BSTRIDE]);
    for (k = 0; k < 3; k++) {
      /*p = in[i*OD_FILT_BSTRIDE + offset] - row*/;
      p = _mm_sub_epi16(_mm_loadl_epi64((__m128i*)&in[i*OD_FILT_BSTRIDE +
       OD_DIRECTION_OFFSETS_TABLE[dir][k]]), row);
      /*if (abs(p) < thresh) sum += taps[k]*p*/
      cmp = od_cmplt_abs_epi16(p, thresh);
      p = _mm_mullo_epi16(p, _mm_set1_epi16(taps[k]));
      p = _mm_and_si128(p, cmp);
      sum = _mm_add_epi16(sum, p);
      /*p = in[i*OD_FILT_BSTRIDE - offset] - row*/;
      p = _mm_sub_epi16(_mm_loadl_epi64((__m128i*)&in[i*OD_FILT_BSTRIDE -
       OD_DIRECTION_OFFSETS_TABLE[dir][k]]), row);
      /*if (abs(p) < thresh) sum += taps[k]*p1*/
      cmp = od_cmplt_abs_epi16(p, thresh);
      p = _mm_mullo_epi16(p, _mm_set1_epi16(taps[k]));
      p = _mm_and_si128(p, cmp);
      sum = _mm_add_epi16(sum, p);
    }
    /*res = row + ((sum + 8) >> 4)*/
    res = _mm_add_epi16(sum, _mm_set1_epi16(8));
    res = _mm_srai_epi16(res, 4);
    res = _mm_add_epi16(row, res);
    _mm_storel_epi64((__m128i*)&y[i*ystride], res);
  }
}
#endif

/* Smooth in the direction detected. */
void od_filter_dering_direction_c(int16_t *y, int ystride, const int16_t *in,
                                  int ln, int threshold, int dir) {
  int i;
  int j;
  int k;
  static const int taps[3] = { 3, 2, 1 };
  for (i = 0; i < 1 << ln; i++) {
    for (j = 0; j < 1 << ln; j++) {
      int16_t sum;
      int16_t xx;
      int16_t yy;
      xx = in[i * OD_FILT_BSTRIDE + j];
      sum = 0;
      for (k = 0; k < 3; k++) {
        int16_t p0;
        int16_t p1;
        p0 = in[i * OD_FILT_BSTRIDE + j + OD_DIRECTION_OFFSETS_TABLE[dir][k]] -
             xx;
        p1 = in[i * OD_FILT_BSTRIDE + j - OD_DIRECTION_OFFSETS_TABLE[dir][k]] -
             xx;
        if (abs(p0) < threshold) sum += taps[k] * p0;
        if (abs(p1) < threshold) sum += taps[k] * p1;
      }
      yy = xx + ((sum + 8) >> 4);
      y[i * ystride + j] = yy;
    }
  }
}

void od_filter_dering_direction_4x4_c(int16_t *y, int ystride,
                                      const int16_t *in, int threshold,
                                      int dir) {
#ifdef ENABLE_SSE2
  od_filter_dering_direction_4x4_sse2(y, ystride, in, threshold, dir);
#else
  od_filter_dering_direction_c(y, ystride, in, 2, threshold, dir);
#endif
}

void od_filter_dering_direction_8x8_c(int16_t *y, int ystride,
                                      const int16_t *in, int threshold,
                                      int dir) {
#ifdef ENABLE_SSE2
  od_filter_dering_direction_8x8_sse2(y, ystride, in, threshold, dir);
#else
  od_filter_dering_direction_c(y, ystride, in, 3, threshold, dir);
#endif
}

#ifdef ENABLE_SSE2

void od_filter_dering_orthogonal_4x4_sse2(int16_t *y, int ystride,
 const int16_t *in, const int16_t *x, int xstride, int threshold, int dir) {
  int i;
  int k;
  int offset;
  __m128i res;
  __m128i p;
  __m128i cmp;
  __m128i row;
  __m128i sum;
  __m128i thresh;
  if (dir > 0 && dir < 4) offset = OD_FILT_BSTRIDE;
  else offset = 1;
  for (i = 0; i < 4; i++) {
    sum = _mm_set1_epi16(0);
    row = _mm_loadl_epi64((__m128i*)&in[i*OD_FILT_BSTRIDE]);
    /*thresh = OD_MINI(threshold, threshold/3
       + abs(in[i*OD_FILT_BSTRIDE] - x[i*xstride]))*/
    thresh = _mm_min_epi16(_mm_set1_epi16(threshold),
     _mm_add_epi16(_mm_set1_epi16(threshold/3),
     od_abs_epi16(_mm_sub_epi16(row,
     _mm_loadl_epi64((__m128i*)&x[i*xstride])))));
    for (k = 1; k <= 2; k++) {
      /*p = in[i*OD_FILT_BSTRIDE + k*offset] - row*/
      p = _mm_sub_epi16(_mm_loadl_epi64((__m128i*)&in[i*OD_FILT_BSTRIDE +
       k*offset]), row);
      /*if (abs(p) < thresh) sum += p*/
      cmp = od_cmplt_abs_epi16(p, thresh);
      p = _mm_and_si128(p, cmp);
      sum = _mm_add_epi16(sum, p);
      /*p = in[i*OD_FILT_BSTRIDE - k*offset] - row*/
      p = _mm_sub_epi16(_mm_loadl_epi64((__m128i*)&in[i*OD_FILT_BSTRIDE -
       k*offset]), row);
      /*if (abs(p) < thresh) sum += p*/
      cmp = od_cmplt_abs_epi16(p, thresh);
      p = _mm_and_si128(p, cmp);
      sum = _mm_add_epi16(sum, p);
    }
    /*row + ((3*sum + 8) >> 4)*/
    res = _mm_mullo_epi16(sum, _mm_set1_epi16(3));
    res = _mm_add_epi16(res, _mm_set1_epi16(8));
    res = _mm_srai_epi16(res, 4);
    res = _mm_add_epi16(res, row);
    _mm_storel_epi64((__m128i*)&y[i*ystride], res);
  }
}

void od_filter_dering_orthogonal_8x8_sse2(int16_t *y, int ystride,
 const int16_t *in, const int16_t *x, int xstride, int threshold, int dir) {
  int i;
  int k;
  int offset;
  __m128i res;
  __m128i p;
  __m128i cmp;
  __m128i row;
  __m128i sum;
  __m128i thresh;
  if (dir > 0 && dir < 4) offset = OD_FILT_BSTRIDE;
  else offset = 1;
  for (i = 0; i < 8; i++) {
    sum = _mm_set1_epi16(0);
    row = _mm_loadu_si128((__m128i*)&in[i*OD_FILT_BSTRIDE]);
    /*thresh = OD_MINI(threshold, threshold/3
       + abs(in[i*OD_FILT_BSTRIDE] - x[i*xstride]))*/
    thresh = _mm_min_epi16(_mm_set1_epi16(threshold),
     _mm_add_epi16(_mm_set1_epi16(threshold/3),
     od_abs_epi16(_mm_sub_epi16(row,
     _mm_loadu_si128((__m128i*)&x[i*xstride])))));
    for (k = 1; k <= 2; k++) {
      /*p = in[i*OD_FILT_BSTRIDE + k*offset] - row*/
      p = _mm_sub_epi16(_mm_loadu_si128((__m128i*)&in[i*OD_FILT_BSTRIDE +
       k*offset]), row);
      /*if (abs(p) < thresh) sum += p*/
      cmp = od_cmplt_abs_epi16(p, thresh);
      p = _mm_and_si128(p, cmp);
      sum = _mm_add_epi16(sum, p);
      /*p = in[i*OD_FILT_BSTRIDE - k*offset] - row*/
      p = _mm_sub_epi16(_mm_loadu_si128((__m128i*)&in[i*OD_FILT_BSTRIDE -
       k*offset]), row);
      /*if (abs(p) < thresh) sum += p*/
      cmp = od_cmplt_abs_epi16(p, thresh);
      p = _mm_and_si128(p, cmp);
      sum = _mm_add_epi16(sum, p);
    }
    /*row + ((3*sum + 8) >> 4)*/
    res = _mm_mullo_epi16(sum, _mm_set1_epi16(3));
    res = _mm_add_epi16(res, _mm_set1_epi16(8));
    res = _mm_srai_epi16(res, 4);
    res = _mm_add_epi16(res, row);
    _mm_storeu_si128((__m128i*)&y[i*ystride], res);
  }
}
#endif

/* Smooth in the direction orthogonal to what was detected. */
void od_filter_dering_orthogonal_c(int16_t *y, int ystride, const int16_t *in,
                                   const od_dering_in *x, int xstride, int ln,
                                   int threshold, int dir) {
  int i;
  int j;
  int offset;
  if (dir > 0 && dir < 4)
    offset = OD_FILT_BSTRIDE;
  else
    offset = 1;
  for (i = 0; i < 1 << ln; i++) {
    for (j = 0; j < 1 << ln; j++) {
      int16_t athresh;
      int16_t yy;
      int16_t sum;
      int16_t p;
      /* Deringing orthogonal to the direction uses a tighter threshold
         because we want to be conservative. We've presumably already
         achieved some deringing, so the amount of change is expected
         to be low. Also, since we might be filtering across an edge, we
         want to make sure not to blur it. That being said, we might want
         to be a little bit more aggressive on pure horizontal/vertical
         since the ringing there tends to be directional, so it doesn't
         get removed by the directional filtering. */
      athresh = OD_MINI(
          threshold, threshold / 3 +
                         abs(in[i * OD_FILT_BSTRIDE + j] - x[i * xstride + j]));
      yy = in[i * OD_FILT_BSTRIDE + j];
      sum = 0;
      p = in[i * OD_FILT_BSTRIDE + j + offset] - yy;
      if (abs(p) < athresh) sum += p;
      p = in[i * OD_FILT_BSTRIDE + j - offset] - yy;
      if (abs(p) < athresh) sum += p;
      p = in[i * OD_FILT_BSTRIDE + j + 2 * offset] - yy;
      if (abs(p) < athresh) sum += p;
      p = in[i * OD_FILT_BSTRIDE + j - 2 * offset] - yy;
      if (abs(p) < athresh) sum += p;
      y[i * ystride + j] = yy + ((3 * sum + 8) >> 4);
    }
  }
}

void od_filter_dering_orthogonal_4x4_c(int16_t *y, int ystride,
                                       const int16_t *in, const od_dering_in *x,
                                       int xstride, int threshold, int dir) {
#ifdef ENABLE_SSE2
  od_filter_dering_orthogonal_4x4_sse2(y, ystride, in, x, xstride, threshold, dir);
#else
  od_filter_dering_orthogonal_c(y, ystride, in, x, xstride, 2, threshold, dir);
#endif
}

void od_filter_dering_orthogonal_8x8_c(int16_t *y, int ystride,
                                       const int16_t *in, const od_dering_in *x,
                                       int xstride, int threshold, int dir) {
#ifdef ENABLE_SSE2
  od_filter_dering_orthogonal_8x8_sse2(y, ystride, in, x, xstride, threshold, dir);
#else
  od_filter_dering_orthogonal_c(y, ystride, in, x, xstride, 3, threshold, dir);
#endif
}

/* This table approximates x^0.16 with the index being log2(x). It is clamped
   to [-.5, 3]. The table is computed as:
   round(256*min(3, max(.5, 1.08*(sqrt(2)*2.^([0:17]+8)/256/256).^.16))) */
static const int16_t OD_THRESH_TABLE_Q8[18] = {
  128, 134, 150, 168, 188, 210, 234, 262, 292,
  327, 365, 408, 455, 509, 569, 635, 710, 768,
};

/* Compute deringing filter threshold for each 8x8 block based on the
   directional variance difference. A high variance difference means that we
   have a highly directional pattern (e.g. a high contrast edge), so we can
   apply more deringing. A low variance means that we either have a low
   contrast edge, or a non-directional texture, so we want to be careful not
   to blur. */
static void od_compute_thresh(int thresh[OD_DERING_NBLOCKS][OD_DERING_NBLOCKS],
                              int threshold,
                              int32_t var[OD_DERING_NBLOCKS][OD_DERING_NBLOCKS],
                              int nhb, int nvb) {
  int bx;
  int by;
  for (by = 0; by < nvb; by++) {
    for (bx = 0; bx < nhb; bx++) {
      int v1;
      /* We use the variance of 8x8 blocks to adjust the threshold. */
      v1 = OD_MINI(32767, var[by][bx] >> 6);
      thresh[by][bx] = (threshold * OD_THRESH_TABLE_Q8[OD_ILOG(v1)] + 128) >> 8;
    }
  }
}

void od_dering(const od_dering_opt_vtbl *vtbl, int16_t *y, int ystride,
               const od_dering_in *x, int xstride, int nhb, int nvb, int sbx,
               int sby, int nhsb, int nvsb, int xdec,
               int dir[OD_DERING_NBLOCKS][OD_DERING_NBLOCKS], int pli,
               unsigned char *bskip, int skip_stride, int threshold,
               int overlap, int coeff_shift) {
  int i;
  int j;
  int bx;
  int by;
  int16_t inbuf[OD_DERING_INBUF_SIZE];
  int16_t *in;
  int bsize;
  int32_t var[OD_DERING_NBLOCKS][OD_DERING_NBLOCKS];
  int thresh[OD_DERING_NBLOCKS][OD_DERING_NBLOCKS];
  bsize = 3 - xdec;
  in = inbuf + OD_FILT_BORDER * OD_FILT_BSTRIDE + OD_FILT_BORDER;
  /* We avoid filtering the pixels for which some of the pixels to average
     are outside the frame. We could change the filter instead, but it would
     add special cases for any future vectorization. */
  for (i = 0; i < OD_DERING_INBUF_SIZE; i++) inbuf[i] = OD_DERING_VERY_LARGE;
  for (i = -OD_FILT_BORDER * (sby != 0);
       i < (nvb << bsize) + OD_FILT_BORDER * (sby != nvsb - 1); i++) {
    for (j = -OD_FILT_BORDER * (sbx != 0);
         j < (nhb << bsize) + OD_FILT_BORDER * (sbx != nhsb - 1); j++) {
      in[i * OD_FILT_BSTRIDE + j] = x[i * xstride + j];
    }
  }
  /* Assume deringing filter is sparsely applied, so do one large copy rather
     than small copies later if deringing is skipped. */
  for (i = 0; i < nvb << bsize; i++) {
    for (j = 0; j < nhb << bsize; j++) {
      y[i * ystride + j] = in[i * OD_FILT_BSTRIDE + j];
    }
  }
  if (pli == 0) {
    for (by = 0; by < nvb; by++) {
      for (bx = 0; bx < nhb; bx++) {
        dir[by][bx] = od_dir_find8_sse2(&x[8 * by * xstride + 8 * bx], xstride,
                                   &var[by][bx], coeff_shift);
      }
    }
    od_compute_thresh(thresh, threshold, var, nhb, nvb);
  } else {
    for (by = 0; by < nvb; by++) {
      for (bx = 0; bx < nhb; bx++) {
        thresh[by][bx] = threshold;
      }
    }
  }
  for (by = 0; by < nvb; by++) {
    for (bx = 0; bx < nhb; bx++) {
      int skip;
#if defined(DAALA_ODINTRIN)
      int xstart;
      int ystart;
      int xend;
      int yend;
      xstart = ystart = 0;
      xend = yend = (2 >> xdec);
      if (overlap) {
        xstart -= (sbx != 0);
        ystart -= (sby != 0);
        xend += (sbx != nhsb - 1);
        yend += (sby != nvsb - 1);
      }
      skip = 1;
      /* We look at whether the current block and its 4x4 surrounding (due to
         lapping) are skipped to avoid filtering the same content multiple
         times. */
      for (i = ystart; i < yend; i++) {
        for (j = xstart; j < xend; j++) {
          skip = skip && bskip[((by << 1 >> xdec) + i) * skip_stride +
                               (bx << 1 >> xdec) + j];
        }
      }
#else
      (void)overlap;
      skip = bskip[by * skip_stride + bx];
#endif
      if (skip) thresh[by][bx] = 0;
    }
  }
  for (by = 0; by < nvb; by++) {
    for (bx = 0; bx < nhb; bx++) {
      if (thresh[by][bx] == 0) continue;
      (vtbl->filter_dering_direction[bsize - OD_LOG_BSIZE0])(
          &y[(by * ystride << bsize) + (bx << bsize)], ystride,
          &in[(by * OD_FILT_BSTRIDE << bsize) + (bx << bsize)], thresh[by][bx],
          dir[by][bx]);
    }
  }
  for (i = 0; i < nvb << bsize; i++) {
    for (j = 0; j < nhb << bsize; j++) {
      in[i * OD_FILT_BSTRIDE + j] = y[i * ystride + j];
    }
  }
  for (by = 0; by < nvb; by++) {
    for (bx = 0; bx < nhb; bx++) {
      if (thresh[by][bx] == 0) continue;
      (vtbl->filter_dering_orthogonal[bsize - OD_LOG_BSIZE0])(
          &y[(by * ystride << bsize) + (bx << bsize)], ystride,
          &in[(by * OD_FILT_BSTRIDE << bsize) + (bx << bsize)],
          &x[(by * xstride << bsize) + (bx << bsize)], xstride, thresh[by][bx],
          dir[by][bx]);
    }
  }
}
