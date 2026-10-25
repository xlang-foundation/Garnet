// SPDX-License-Identifier: Apache-2.0
// Compile the immutable pre-integration sampler with distinct entry points.
// This supplies the GPU reference even when production dispatch opts in.
#define runLogitsTop1FP32 runReferenceLogitsTop1FP32
#define runLogitsTop1BF16 runReferenceLogitsTop1BF16
#define runLogitsTop1BatchFP32 runReferenceLogitsTop1BatchFP32
#define runLogitsTop1BatchBF16 runReferenceLogitsTop1BatchBF16
#define sampleRepetitionFP32 sampleReferenceRepetitionFP32
#define sampleRepetitionBF16 sampleReferenceRepetitionBF16
#include "evidence/parallel-batch-top1-legacy-sampler.cu.txt"
