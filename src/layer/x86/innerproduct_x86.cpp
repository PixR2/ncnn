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

#include "innerproduct_x86.h"

#if __SSE3__
#include <NEON_2_SSE.h>
#endif // __SSE3__

namespace ncnn {

DEFINE_LAYER_CREATOR(InnerProduct_x86)

int InnerProduct_x86::forward(const Mat& bottom_blob, Mat& top_blob) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int size = w * h;

    top_blob.create(1, 1, num_output);
    if (top_blob.empty())
        return -100;

    // num_output
    const float* weight_data_ptr = weight_data;
    #pragma omp parallel for
    for (int p=0; p<num_output; p++)
    {
        float* outptr = top_blob.channel(p);
        float sum = 0.f;

        if (bias_term)
            sum = bias_data.data[p];

        const float* w = weight_data_ptr + size * channels * p;
        const float* w2 = w + size;

#if __SSE3__
        float32x4_t _sum = vdupq_n_f32(0.f);
        float32x4_t _sum2 = vdupq_n_f32(0.f);
#endif // __SSE3__

        // channels
        for (int q=0; q<channels; q++)
        {
            const float* m = bottom_blob.channel(q);

#if __SSE3__
            int nn = size >> 3;
            int remain = size & 7;
#else
            int remain = size;
#endif // __SSE3__

#if __SSE3__
            for (; nn>0; nn--)
            {
                float32x4_t _m = vld1q_f32(m);
                float32x4_t _w = vld1q_f32(w);
                _sum = vfmaq_f32(_sum, _m, _w);

                _m = vld1q_f32(m + 4);
                _w = vld1q_f32(w + 4);
                _sum2 = vfmaq_f32(_sum2, _m, _w);

                m += 8;
                w += 8;
            }
#endif // __SSE3__
            for (; remain>0; remain--)
            {
                sum += *m * *w;

                m++;
                w++;
            }
        }

#if __SSE3__
        _sum = vaddq_f32(_sum, _sum2);
        sum += vaddvq_f32(_sum);
#endif // __SSE3__

        outptr[0] = sum;
    }

    return 0;
}

} // namespace ncnn
