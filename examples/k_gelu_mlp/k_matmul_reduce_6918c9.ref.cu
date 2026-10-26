extern "C" __global__
__launch_bounds__(8) void k_matmul_reduce_6918c9(const float* gelu_inv_sqrt2, const float* gelu_half, const float* gelu_one, const float* x, const float* w1, const float* w2, float* matmul_1) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0 = _gid / 1024;
        int a1 = (_gid / 8) % 128;
        int a2_co = (_gid) % 8;
        float in0 = gelu_half[0];
        float in1 = gelu_inv_sqrt2[0];
        float in2 = gelu_one[0];
        float acc1 = 0.0f;
        for (int a2 = a2_co; a2 < 512; a2 += 8) {
            float acc0 = 0.0f;
            for (int a3 = 0; a3 < 128; a3++) {
                float in4 = x[a0 * 128 + a3];
                float in3 = w1[a3 * 512 + a2];
                float v0 = in3 * in4;
                acc0 += v0;
            }
            float in5 = w2[a2 * 128 + a1];
            float v6 = in5 * ((in2 + erff((acc0 * in1))) * (acc0 * in0));
            acc1 += v6;
        }
        {
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 4);
            acc1 = acc1 + acc1__o;
        }
        {
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 2);
            acc1 = acc1 + acc1__o;
        }
        {
            float acc1__o = __shfl_xor_sync(__activemask(), acc1, 1);
            acc1 = acc1 + acc1__o;
        }
        if (a2_co == 0) {
            matmul_1[a0 * 128 + a1] = acc1;
        }
}
