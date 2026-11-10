// Garnet entry point over the MLX affine quantized GEMM (the qmm_t kernel MLX and
// vLLM-Metal use for 4-bit checkpoints). Appended after MLX's quantized.h, so the
// MLX loader and simdgroup MMA templates are in scope. FP32 activations are rounded
// to BF16 while staging, matching MLX running a BF16 checkpoint; accumulation and
// the stored result stay FP32 with Garnet's row strides. The host guarantees
// 16-byte aligned activation rows.
struct GarnetAffineQmmParams {
  ulong rows;
  ulong inner;
  ulong columns;
  ulong aStride;
  ulong oStride;
};

template <int group_size, int bits, int BM, int BK, int BN>
[[kernel]] void garnet_affine_qmm_bf16(
    const device float* x [[buffer(0)]],
    const device uint32_t* w [[buffer(1)]],
    const device bfloat16_t* scales [[buffer(2)]],
    const device bfloat16_t* biases [[buffer(3)]],
    device float* y [[buffer(4)]],
    const constant GarnetAffineQmmParams& p [[buffer(5)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  using T = bfloat16_t;
  constexpr int WM = 2;
  constexpr int WN = 2;
  constexpr int threads = WM * WN * SIMD_SIZE;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = BK + 16 / sizeof(T);
  constexpr int x_per_thread = BM * BK / threads;
  constexpr int x_vectors = x_per_thread / 4;
  constexpr int x_threads_per_row = BK / x_per_thread;
  static_assert(BM * BK % threads == 0 && BK % x_per_thread == 0 && x_per_thread % 4 == 0,
      "x tile must split into float4 runs");

  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];

  using mma_t = mlx::steel::
      BlockMMA<T, float, BM, BN, BK, WM, WN, false, true, BK_padded, BK_padded>;
  using loader_w_t = QuantizedBlockLoader<T, BN, BK, BK_padded, 1, threads, group_size, bits>;

  const int K = int(p.inner);
  const int N = int(p.columns);
  const int M = int(p.rows);
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w + int64_t(y_col) * K_w;
  scales += int64_t(y_col) * K_g;
  biases += int64_t(y_col) * K_g;

  const short num_els = min(BM, M - y_row);
  const short num_outs = min(BN, N - y_col);
  const short x_row = lid / x_threads_per_row;
  const short x_col = (lid % x_threads_per_row) * x_per_thread;
  const device float4* xr =
      (const device float4*)(x + int64_t(y_row + x_row) * int64_t(p.aStride) + x_col);
  threadgroup T* xs = Xs + x_row * BK_padded + x_col;
  const bool x_live = x_row < num_els;

  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);
  mma_t mma_op(simd_gid, simd_lid);

  for (int k = 0; k < K; k += BK) {
    float4 xv[x_vectors];
    if (x_live) {
      for (int i = 0; i < x_vectors; i++) xv[i] = xr[k / 4 + i];
    } else {
      for (int i = 0; i < x_vectors; i++) xv[i] = float4(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < x_vectors; i++) {
      xs[4 * i + 0] = T(xv[i].x);
      xs[4 * i + 1] = T(xv[i].y);
      xs[4 * i + 2] = T(xv[i].z);
      xs[4 * i + 3] = T(xv[i].w);
    }
    if (num_outs < BN) {
      loader_w.load_safe(short2(BK, num_outs));
    } else {
      loader_w.load_unsafe();
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(Xs, Ws);
    loader_w.next();
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  device float* out = y + int64_t(y_row) * int64_t(p.oStride) + y_col;
  if (num_els < BM || num_outs < BN) {
    mma_op.store_result_safe(out, int(p.oStride), short2(num_outs, num_els));
  } else {
    mma_op.store_result(out, int(p.oStride));
  }
}

#define GARNET_QMM_INSTANCE(BM, BK, BN)                                                     \
  template [[host_name("garnet_affine_qmm_bf16_gs64_b4_bm" #BM "_bk" #BK "_bn" #BN)]]      \
  [[kernel]] decltype(garnet_affine_qmm_bf16<64, 4, BM, BK, BN>)                            \
      garnet_affine_qmm_bf16<64, 4, BM, BK, BN>;

GARNET_QMM_INSTANCE(32, 32, 32)
GARNET_QMM_INSTANCE(64, 32, 64)
