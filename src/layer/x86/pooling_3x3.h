// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#if __SSE3__
#include <NEON_2_SSE.h>
#endif // __SSE3__

static void pooling3x3s2_max_sse(const Mat& bottom_blob, Mat& top_blob)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2*outw + w;

    #pragma omp parallel for
    for (int q=0; q<inch; q++)
    {
        const float* img0 = bottom_blob.channel(q);
        float* outptr = top_blob.channel(q);

        const float* r0 = img0;
        const float* r1 = img0 + w;
        const float* r2 = img0 + w*2;

        for (int i = 0; i < outh; i++)
        {
#if __SSE3__
            int nn = outw >> 2;
            int remain = outw - (nn << 2);
#else
            int remain = outw;
#endif // __SSE3__

#if __SSE3__
            float32x4x2_t _r0 = vld2q_f32(r0);
            float32x4x2_t _r1 = vld2q_f32(r1);
            float32x4x2_t _r2 = vld2q_f32(r2);
            for (; nn>0; nn--)
            {
                float32x4x2_t _r0n = vld2q_f32(r0+8);
                float32x4x2_t _r1n = vld2q_f32(r1+8);
                float32x4x2_t _r2n = vld2q_f32(r2+8);

                float32x4_t _max0 = vmaxq_f32(_r0.val[0], _r0.val[1]);
                float32x4_t _max1 = vmaxq_f32(_r1.val[0], _r1.val[1]);
                float32x4_t _max2 = vmaxq_f32(_r2.val[0], _r2.val[1]);

                float32x4_t _r02 = vextq_f32(_r0.val[0], _r0n.val[0], 1);
                float32x4_t _r12 = vextq_f32(_r1.val[0], _r1n.val[0], 1);
                float32x4_t _r22 = vextq_f32(_r2.val[0], _r2n.val[0], 1);

                _max0 = vmaxq_f32(_max0, _r02);
                _max1 = vmaxq_f32(_max1, _r12);
                _max2 = vmaxq_f32(_max2, _r22);

                float32x4_t _max = vmaxq_f32(vmaxq_f32(_max0, _max1), _max2);

                vst1q_f32(outptr, _max);

                _r0 = _r0n;
                _r1 = _r1n;
                _r2 = _r2n;

                r0 += 8;
                r1 += 8;
                r2 += 8;
                outptr += 4;
            }
#endif // __SSE3__
            for (; remain>0; remain--)
            {
                float max0 = std::max(std::max(r0[0], r0[1]), r0[2]);
                float max1 = std::max(std::max(r1[0], r1[1]), r1[2]);
                float max2 = std::max(std::max(r2[0], r2[1]), r2[2]);

                *outptr = std::max(std::max(max0, max1), max2);

                r0 += 2;
                r1 += 2;
                r2 += 2;
                outptr++;
            }

            r0 += tailstep;//1 + w;
            r1 += tailstep;//1 + w;
            r2 += tailstep;//1 + w;
        }
    }
}
