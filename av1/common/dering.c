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
#include "aom/aom_integer.h"
#include "av1/common/dering.h"
#include "av1/common/onyxc_int.h"
#include "av1/common/reconinter.h"
#include "av1/common/od_dering.h"

int compute_level_from_index(int global_level, int gi) {
  static const int dering_gains[DERING_REFINEMENT_LEVELS] = { 0, 11, 16, 22 };
  int level;
  if (global_level == 0) return 0;
  level = (global_level * dering_gains[gi] + 8) >> 4;
  return clamp(level, gi, MAX_DERING_LEVEL - 1);
}

int sb_all_skip(const AV1_COMMON *const cm, int mi_row, int mi_col) {
  int r, c;
  int maxc, maxr;
  int skip = 1;
  maxc = cm->mi_cols - mi_col;
  maxr = cm->mi_rows - mi_row;
  if (maxr > MAX_MIB_SIZE) maxr = MAX_MIB_SIZE;
  if (maxc > MAX_MIB_SIZE) maxc = MAX_MIB_SIZE;
  for (r = 0; r < maxr; r++) {
    for (c = 0; c < maxc; c++) {
      skip = skip &&
             cm->mi_grid_visible[(mi_row + r) * cm->mi_stride + mi_col + c]
                 ->mbmi.skip;
    }
  }
  return skip;
}

int sb_all_skip_out(const AV1_COMMON *const cm, int mi_row, int mi_col,
                    int *bskip) {
  int r, c;
  int maxc, maxr;
  int skip = 1;
  MODE_INFO **grid;
  grid = cm->mi_grid_visible;
  maxc = cm->mi_cols - mi_col;
  maxr = cm->mi_rows - mi_row;
  if (maxr > MAX_MIB_SIZE) maxr = MAX_MIB_SIZE;
  if (maxc > MAX_MIB_SIZE) maxc = MAX_MIB_SIZE;
  for (r = 0; r < maxr; r++) {
    MODE_INFO **grid_row;
    grid_row = &grid[(mi_row + r) * cm->mi_stride + mi_col];
    for (c = 0; c < maxc; c++) {
      int tmp;
      tmp = grid_row[c]->mbmi.skip;
      bskip[r*MAX_MIB_SIZE + c] = tmp;
      skip = skip && tmp;
    }
  }
  return skip;
}

#include <emmintrin.h>
static void copy_sb16_8(uint8_t * restrict dst, int dstride, od_dering_in * restrict src,
                        int sstride, int vsize, int hsize)
{
  int r, c;
  if (hsize == 64) {
    for (r = 0; r < vsize; ++r) {
      __m128i tmp;
      tmp = _mm_loadu_si128((__m128i*)&src[0]);
      tmp = _mm_packus_epi16(tmp, tmp);
      _mm_storel_epi64((__m128i*)&dst[0], tmp);

      tmp = _mm_loadu_si128((__m128i*)&src[8]);
      tmp = _mm_packus_epi16(tmp, tmp);
      _mm_storel_epi64((__m128i*)&dst[8], tmp);

      tmp = _mm_loadu_si128((__m128i*)&src[16]);
      tmp = _mm_packus_epi16(tmp, tmp);
      _mm_storel_epi64((__m128i*)&dst[16], tmp);

      tmp = _mm_loadu_si128((__m128i*)&src[24]);
      tmp = _mm_packus_epi16(tmp, tmp);
      _mm_storel_epi64((__m128i*)&dst[24], tmp);

      tmp = _mm_loadu_si128((__m128i*)&src[32]);
      tmp = _mm_packus_epi16(tmp, tmp);
      _mm_storel_epi64((__m128i*)&dst[32], tmp);

      tmp = _mm_loadu_si128((__m128i*)&src[40]);
      tmp = _mm_packus_epi16(tmp, tmp);
      _mm_storel_epi64((__m128i*)&dst[40], tmp);

      tmp = _mm_loadu_si128((__m128i*)&src[48]);
      tmp = _mm_packus_epi16(tmp, tmp);
      _mm_storel_epi64((__m128i*)&dst[48], tmp);

      tmp = _mm_loadu_si128((__m128i*)&src[56]);
      tmp = _mm_packus_epi16(tmp, tmp);
      _mm_storel_epi64((__m128i*)&dst[56], tmp);

      src += sstride;
      dst += dstride;
    }
  } else if (hsize == 32) {
    for (r = 0; r < vsize; ++r) {
      __m128i tmp;
      tmp = _mm_loadu_si128((__m128i*)&src[0]);
      tmp = _mm_packus_epi16(tmp, tmp);
      _mm_storel_epi64((__m128i*)&dst[0], tmp);

      tmp = _mm_loadu_si128((__m128i*)&src[8]);
      tmp = _mm_packus_epi16(tmp, tmp);
      _mm_storel_epi64((__m128i*)&dst[8], tmp);

      tmp = _mm_loadu_si128((__m128i*)&src[16]);
      tmp = _mm_packus_epi16(tmp, tmp);
      _mm_storel_epi64((__m128i*)&dst[16], tmp);

      tmp = _mm_loadu_si128((__m128i*)&src[24]);
      tmp = _mm_packus_epi16(tmp, tmp);
      _mm_storel_epi64((__m128i*)&dst[24], tmp);

      src += sstride;
      dst += dstride;
    }
  } else {
    for (r = 0; r < vsize; ++r) {
      for (c = 0; c < hsize - 7; c+=8) {
        __m128i tmp;
        tmp = _mm_loadu_si128((__m128i*)&src[c]);
        tmp = _mm_packus_epi16(tmp, tmp);
        _mm_storel_epi64((__m128i*)&dst[c], tmp);
      }
      for (; c < hsize; ++c) {
        dst[c] = src[c];
      }
      src += sstride;
      dst += dstride;
    }
  }
}


void av1_dering_frame(YV12_BUFFER_CONFIG *frame, AV1_COMMON *cm,
                      MACROBLOCKD *xd, int global_level) {
  int r, c;
  int sbr, sbc;
  int nhsb, nvsb;
  od_dering_in *dst[3];
  int dir[OD_DERING_NBLOCKS][OD_DERING_NBLOCKS] = { { 0 } };
  int stride;
  int bsize[3];
  int dec[3];
  int pli;
  int coeff_shift = AOMMAX(cm->bit_depth - 8, 0);
  int dst_read, dst_write;
  int toggle = 0;
  int linesize;
  int *sbskip;
  int (*bskip)[MAX_MIB_SIZE*MAX_MIB_SIZE];
  od_dering_in *inbuf[3];
  nvsb = (cm->mi_rows + MAX_MIB_SIZE - 1) / MAX_MIB_SIZE;
  nhsb = (cm->mi_cols + MAX_MIB_SIZE - 1) / MAX_MIB_SIZE;
  av1_setup_dst_planes(xd->plane, frame, 0, 0);
  for (pli = 0; pli < 3; pli++) {
    dec[pli] = xd->plane[pli].subsampling_x;
    bsize[pli] = 8 >> dec[pli];
  }
  stride = bsize[0] * cm->mi_cols;
  for (pli = 0; pli < 3; pli++) {
    dst[pli] = aom_malloc(sizeof(*dst) * cm->mi_rows * (cm->mi_cols+7) * 64);
    inbuf[pli] = aom_malloc(sizeof(inbuf[0][0]) * nhsb * OD_DERING_INBUF_SIZE *
        2);
  }
  sbskip = aom_malloc(sizeof(*sbskip) * nhsb * 2);
  bskip = aom_malloc(sizeof(*bskip) * nhsb * 2);
  linesize = cm->mi_rows * (cm->mi_cols+7) * 64/2;
  for (sbr = 0; sbr < nvsb; sbr++) {
    toggle = sbr&1;
    dst_write = 0;

    for (sbc = 0; sbc < nhsb; sbc++) {
      int level;
      int nhb, nvb;
      nhb = AOMMIN(MAX_MIB_SIZE, cm->mi_cols - MAX_MIB_SIZE * sbc);
      nvb = AOMMIN(MAX_MIB_SIZE, cm->mi_rows - MAX_MIB_SIZE * sbr);
      sbskip[toggle*nhsb + sbc] =
          sb_all_skip_out(cm, sbr * MAX_MIB_SIZE, sbc * MAX_MIB_SIZE,
                      bskip[toggle*nhsb + sbc]);
      if (sbskip[toggle*nhsb + sbc]) continue;
      for (pli = 0; pli < 3; pli++) {
        int threshold;
        level = compute_level_from_index(
            global_level,
            cm->mi_grid_visible[MAX_MIB_SIZE * sbr * cm->mi_stride +
                                MAX_MIB_SIZE * sbc]
                ->mbmi.dering_gain);
        /* FIXME: This is a temporary hack that uses more conservative
           deringing for chroma. */
        if (pli) level = (level * 5 + 4) >> 3;
        threshold = level << coeff_shift;
        od_inbuf_copy(&inbuf[pli][(toggle * nhsb + sbc) * OD_DERING_INBUF_SIZE],
            &xd->plane[pli].dst.buf[sbr * bsize[pli] * MAX_MIB_SIZE * xd->plane[pli].dst.stride + sbc * bsize[pli] * MAX_MIB_SIZE],
                              xd->plane[pli].dst.stride,
                              3 - dec[pli], nvb, nhb, sbr, sbc, nvsb, nhsb);
        od_dering(&OD_DERING_VTBL_C,
                  &dst[pli][dst_write + toggle*linesize],
                  OD_BSIZE_MAX,
                  &inbuf[pli][(toggle * nhsb + sbc) * OD_DERING_INBUF_SIZE],
                  nhb, nvb, sbc, sbr, nhsb, nvsb, dec[pli], dir, pli,
                  bskip[toggle*nhsb + sbc],
                  MAX_MIB_SIZE, threshold, coeff_shift);
        dst_write += OD_BSIZE_MAX*OD_BSIZE_MAX;
      }
    }

    if (sbr<1) continue;
    dst_read = 0;
    for (sbc = 0; sbc < nhsb; sbc++) {
      int nhb, nvb;
      nhb = AOMMIN(MAX_MIB_SIZE, cm->mi_cols - MAX_MIB_SIZE * sbc);
      nvb = AOMMIN(MAX_MIB_SIZE, cm->mi_rows - MAX_MIB_SIZE * (sbr-1));
      if (sbskip[(1-toggle)*nhsb + sbc]) continue;
      for (pli = 0; pli < 3; pli++) {
        copy_sb16_8(&xd->plane[pli].dst.buf[xd->plane[pli].dst.stride *
                                         (bsize[pli] * MAX_MIB_SIZE * (sbr-1)) +
                                     sbc * bsize[pli] * MAX_MIB_SIZE],
                    xd->plane[pli].dst.stride,
                    &dst[pli][dst_read + (!toggle)*linesize],
                    OD_BSIZE_MAX,
                    bsize[pli] * nvb, bsize[pli] * nhb);
        dst_read += OD_BSIZE_MAX*OD_BSIZE_MAX;
      }
    }

  }
  /* Copy deringed blocks back. */
  for (sbr = nvsb-1; sbr < nvsb; sbr++) {
    dst_read = 0;
    for (sbc = 0; sbc < nhsb; sbc++) {
      int nhb, nvb;
      nhb = AOMMIN(MAX_MIB_SIZE, cm->mi_cols - MAX_MIB_SIZE * sbc);
      nvb = AOMMIN(MAX_MIB_SIZE, cm->mi_rows - MAX_MIB_SIZE * sbr);
      for (pli = 0; pli < 3; pli++) {
        if (sbskip[toggle*nhsb + sbc]) continue;
        copy_sb16_8(&xd->plane[pli].dst.buf[xd->plane[pli].dst.stride *
                                         (bsize[pli] * MAX_MIB_SIZE * sbr) +
                                     sbc * bsize[pli] * MAX_MIB_SIZE],
                    xd->plane[pli].dst.stride,
                    &dst[pli][dst_read + toggle*linesize],
                    OD_BSIZE_MAX,
                    bsize[pli] * nvb, bsize[pli] * nhb);
        dst_read += OD_BSIZE_MAX*OD_BSIZE_MAX;
      }
    }
  }
  for (pli = 0; pli < 3; pli++) {
    aom_free(dst[pli]);
    aom_free(inbuf[pli]);
  }
  aom_free(bskip);
  aom_free(sbskip);
}
