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
  int global_level;
  int *dering_level;
  dering_level = malloc((cm->mi_rows/MI_BLOCK_SIZE)*(cm->mi_cols/MI_BLOCK_SIZE)*sizeof(int));
  vpx_yv12_copy_y(frame, frame_uf);
  global_level = vp10_dering_search(frame, ref, cm, xd, dering_level);
  vp10_dering_frame(frame, cm, xd, global_level, dering_level);
  // fprintf(stderr, "level %d err %"PRId64"\n", level, err);
  vpx_yv12_copy_y(frame_uf, frame);
  free(dering_level);
  return global_level;
}
