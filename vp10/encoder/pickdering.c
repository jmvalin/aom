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
#include "vp10/common/dering.h"
#include "vp10/common/onyxc_int.h"
#include "vp10/common/reconinter.h"
#include "vp10/encoder/encoder.h"
#include "vpx/vpx_integer.h"

int vp10_try_dering_frame(YV12_BUFFER_CONFIG *frame,
                          YV12_BUFFER_CONFIG *frame_uf,
                          const YV12_BUFFER_CONFIG *ref, VP10_COMMON *cm,
                          MACROBLOCKD *xd) {
  int level;
  int64_t best_error;
  int best_level = 0;
  int *dering_level;
  dering_level = malloc((cm->mi_rows/MI_BLOCK_SIZE)*(cm->mi_cols/MI_BLOCK_SIZE)*sizeof(int));
  vpx_yv12_copy_y(frame, frame_uf);
  best_error = vp10_get_y_sse(ref, frame);
  for (level = 1; level < MAX_DERING_LEVEL; ++level) {
    int64_t error;
    vp10_dering_frame(frame, cm, xd, level);
    error = vp10_get_y_sse(ref, frame);
    if (error < best_error) {
      best_error = error;
      best_level = level;
    }
    // fprintf(stderr, "level %d err %"PRId64"\n", level, err);
    vpx_yv12_copy_y(frame_uf, frame);
  }
  vp10_dering_search(frame, cm, xd, dering_level);
  free(dering_level);
  return best_level;
}
