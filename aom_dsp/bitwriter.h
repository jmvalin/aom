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

#ifndef AOM_DSP_BITWRITER_H_
#define AOM_DSP_BITWRITER_H_

#include "aom_dsp/dkboolwriter.h"
#include "aom_dsp/prob.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct aom_dk_writer aom_writer;

static INLINE void aom_start_encode(aom_writer *bc, uint8_t *buffer) {
  aom_dk_start_encode(bc, buffer);
}

static INLINE void aom_stop_encode(aom_writer *bc) { aom_dk_stop_encode(bc); }

static INLINE void aom_write(aom_writer *br, int bit, int probability) {
  aom_dk_write(br, bit, probability);
}

static INLINE void aom_write_bit(aom_writer *w, int bit) {
  aom_write(w, bit, 128);  // aom_prob_half
}

static INLINE void aom_write_literal(aom_writer *w, int data, int bits) {
  int bit;

  for (bit = bits - 1; bit >= 0; bit--) aom_write_bit(w, 1 & (data >> bit));
}

static INLINE void aom_write_tree_bits(aom_writer *w, const aom_tree_index *tr,
                                       const aom_prob *probs, int bits, int len,
                                       aom_tree_index i) {
  do {
    const int bit = (bits >> --len) & 1;
    aom_write(w, bit, probs[i >> 1]);
    i = tr[i + bit];
  } while (len);
}

static INLINE void aom_write_tree(aom_writer *w, const aom_tree_index *tree,
                                  const aom_prob *probs, int bits, int len,
                                  aom_tree_index i) {
  aom_write_tree_bits(w, tree, probs, bits, len, i);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_DSP_BITWRITER_H_