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
#ifndef AV1_COMMON_CDEF_CONFIG_H_
#define AV1_COMMON_CDEF_CONFIG_H_

#define CDEF_STRENGTH_BITS 7

#define DERING_STRENGTHS 32
#define CLPF_STRENGTHS 4

#define CDEF_MAX_STRENGTHS 16


typedef struct {
  int dering_damping;
  int clpf_damping;
  int nb_strengths;
  int strengths[CDEF_MAX_STRENGTHS];
  int uv_strengths[CDEF_MAX_STRENGTHS];
  int bits;
} CDEFConfig;


#endif
