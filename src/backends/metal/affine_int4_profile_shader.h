// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace Garnet {
// Prefix every affine source. A compiled profile fixes model dimensions and
// storage strides, while actual token/selected-row counts remain runtime data.
// Decode already has the bounded one-to-four-row function-constant variants.
inline constexpr const char* kAffineInt4ProfilePrelude=R"METAL(
#if defined(GARNET_AFFINE_INDEX32) && !defined(GARNET_AFFINE_PROFILE)
#error Bounded_indexes_require_specialized_profiles
#endif
#ifdef GARNET_AFFINE_INDEX32
using AffineIndex=uint;
static_assert(sizeof(AffineIndex)==4,"Compiled affine index width differs");
struct BoundedAffineParams {uint rows,inner,columns,aStride,oStride;};
#else
using AffineIndex=ulong;
#endif
#ifdef GARNET_AFFINE_PROFILE
constant uint affine_profile_inner [[function_constant(10)]];
constant uint affine_profile_columns [[function_constant(11)]];
constant uint affine_profile_a_stride [[function_constant(12)]];
constant uint affine_profile_o_stride [[function_constant(13)]];
#ifdef GARNET_AFFINE_INDEX32
#define GARNET_AFFINE_PARAMETERS(TYPE) const BoundedAffineParams p{uint(runtime.rows),affine_profile_inner,affine_profile_columns,affine_profile_a_stride,affine_profile_o_stride}
#else
#define GARNET_AFFINE_PARAMETERS(TYPE) const TYPE p{runtime.rows,affine_profile_inner,affine_profile_columns,affine_profile_a_stride,affine_profile_o_stride}
#endif
#else
#define GARNET_AFFINE_PARAMETERS(TYPE) constant TYPE& p=runtime
#endif
)METAL";
}
