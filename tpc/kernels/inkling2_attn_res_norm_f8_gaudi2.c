// SPDX-License-Identifier: Apache-2.0
// Fused Inkling2 attn_res + RMSNorm -> (bf16, fp8, fp8_scale) on Gaudi2 TPC.
#define HAS_NORM 1
#define OUTPUT_F8 1
#include "inkling2_attn_res_common.h"
