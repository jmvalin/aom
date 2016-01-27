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

#define VPX_DERING_LEVEL 34

void vp10_dering_frame(YV12_BUFFER_CONFIG *frame, VP10_COMMON *cm,
                       MACROBLOCKD *xd, int level);
#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // VP10_COMMON_DERING_H_
