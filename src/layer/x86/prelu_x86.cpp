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

#include "prelu_x86.h"

#if __SSE3__
#include <NEON_2_SSE.h>
#endif // __SSE3__

namespace ncnn {

DEFINE_LAYER_CREATOR(PReLU_x86)

int PReLU_x86::forward(const Mat& bottom_blob, Mat& top_blob) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int size = w * h;

    top_blob.create(w, h, channels);
    if (top_blob.empty())
        return -100;

    const float* slope_data_ptr = slope_data;

    #pragma omp parallel for
    for (int q=0; q<channels; q++)
    {
        const float* ptr = bottom_blob.channel(q);
        float* outptr = top_blob.channel(q);
        float slope = num_slope > 1 ? slope_data_ptr[q] : slope_data_ptr[0];

#if __SSE3__
        int nn = size >> 2;
        int remain = size - (nn << 2);
#else
        int remain = size;
#endif // __SSE3__

#if __SSE3__
        float32x4_t _zero = vdupq_n_f32(0.f);
        float32x4_t _slope = vdupq_n_f32(slope);
        for (; nn>0; nn--)
        {
            float32x4_t _p = vld1q_f32(ptr);
            uint32x4_t _lemask = vcleq_f32(_p, _zero);
            float32x4_t _ps = vmulq_f32(_p, _slope);
            float32x4_t _outp = vbslq_f32(_lemask, _ps, _p);
            vst1q_f32(outptr, _outp);

            ptr += 4;
            outptr += 4;
        }
#endif // __SSE3__
        for (; remain>0; remain--)
        {
            if (*ptr < 0)
                *outptr = *ptr * slope;
            else
                *outptr = *ptr;

            ptr++;
            outptr++;
        }
    }

    return 0;
}

int PReLU_x86::forward_inplace(Mat& bottom_top_blob) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    const float* slope_data_ptr = slope_data;

    #pragma omp parallel for
    for (int q=0; q<channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);
        float slope = num_slope > 1 ? slope_data_ptr[q] : slope_data_ptr[0];

#if __SSE3__
        int nn = size >> 2;
        int remain = size - (nn << 2);
#else
        int remain = size;
#endif // __SSE3__

#if __SSE3__
        float32x4_t _zero = vdupq_n_f32(0.f);
        float32x4_t _slope = vdupq_n_f32(slope);
        for (; nn>0; nn--)
        {
            float32x4_t _p = vld1q_f32(ptr);
            uint32x4_t _lemask = vcleq_f32(_p, _zero);
            float32x4_t _ps = vmulq_f32(_p, _slope);
            _p = vbslq_f32(_lemask, _ps, _p);
            vst1q_f32(ptr, _p);

            ptr += 4;
        }
#endif // __SSE3__
        for (; remain>0; remain--)
        {
            if (*ptr < 0)
                *ptr *= slope;

            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn
