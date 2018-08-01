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

#include "batchnorm_x86.h"

#if __SSE3__
#include "NEON_2_SSE.h"
#endif // __SSE3__

namespace ncnn {

DEFINE_LAYER_CREATOR(BatchNorm_x86)

int BatchNorm_x86::forward(const Mat& bottom_blob, Mat& top_blob) const
{
    // a = bias - slope * mean / sqrt(var)
    // b = slope / sqrt(var)
    // value = b * value + a

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int size = w * h;

    top_blob.create(w, h, channels);
    if (top_blob.empty())
        return -100;

    const float* a_data_ptr = a_data;
    const float* b_data_ptr = b_data;
    #pragma omp parallel for
    for (int q=0; q<channels; q++)
    {
        const float* ptr = bottom_blob.channel(q);
        float* outptr = top_blob.channel(q);

        float a = a_data_ptr[q];
        float b = b_data_ptr[q];

#if __SSE3__
        int nn = size >> 2;
        int remain = size - (nn << 2);
#else
        int remain = size;
#endif // __SSE3__

#if __SSE3__
        float32x4_t _a = vdupq_n_f32(a);
        float32x4_t _b = vdupq_n_f32(b);
        for (; nn>0; nn--)
        {
            float32x4_t _p = vld1q_f32(ptr);
            float32x4_t _outp = _a;
            _outp = vmlaq_f32(_outp, _p, _b);
            vst1q_f32(outptr, _outp);

            ptr += 4;
            outptr += 4;
        }
#endif // __SSE3__
        for (; remain>0; remain--)
        {
            *outptr = b * *ptr + a;

            ptr++;
            outptr++;
        }
    }

    return 0;
}

int BatchNorm_x86::forward_inplace(Mat& bottom_top_blob) const
{
    // a = bias - slope * mean / sqrt(var)
    // b = slope / sqrt(var)
    // value = b * value + a

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int size = w * h;

    const float* a_data_ptr = a_data;
    const float* b_data_ptr = b_data;
    #pragma omp parallel for
    for (int q=0; q<channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        float a = a_data_ptr[q];
        float b = b_data_ptr[q];

#if __SSE3__
        int nn = size >> 2;
        int remain = size - (nn << 2);
#else
        int remain = size;
#endif // __SSE3__

#if __SSE3__
        float32x4_t _a = vdupq_n_f32(a);
        float32x4_t _b = vdupq_n_f32(b);
        for (; nn>0; nn--)
        {
            float32x4_t _p = vld1q_f32(ptr);
            float32x4_t _outp = _a;
            _outp = vmlaq_f32(_outp, _p, _b);
            vst1q_f32(ptr, _outp);

            ptr += 4;
        }
#endif // __SSE3__
        for (; remain>0; remain--)
        {
            *ptr = b * *ptr + a;

            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn
