#ifndef VP10_COMMON_DERING_H_
#define VP10_COMMON_DERING_H_

#include "vp10/common/od_dering.h"
#include "vp10/common/onyxc_int.h"
#include "vpx/vpx_integer.h"
#include "vpx_config.h"
#include "vpx_ports/mem.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DERING_LEVEL_BITS 6
#define MAX_DERING_LEVEL (1 << DERING_LEVEL_BITS)

#define DERING_REFINEMENT 1

extern double dering_gains[4];
int sb_all_skip(const VP10_COMMON *const cm, int mi_row, int mi_col);
void vp10_dering_frame(YV12_BUFFER_CONFIG *frame, VP10_COMMON *cm,
                       MACROBLOCKD *xd, int global_level);

int vp10_dering_search(YV12_BUFFER_CONFIG *frame, const YV12_BUFFER_CONFIG *ref,
                      VP10_COMMON *cm,
                      MACROBLOCKD *xd);

int vp10_try_dering_frame(YV12_BUFFER_CONFIG *frame,
                          YV12_BUFFER_CONFIG *frame_uf,
                          const YV12_BUFFER_CONFIG *ref,
                          VP10_COMMON *cm,
                          MACROBLOCKD *xd);
#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // VP10_COMMON_DERING_H_
