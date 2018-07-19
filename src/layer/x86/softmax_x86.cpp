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

#include "softmax_x86.h"
#include <float.h>
#include <math.h>

#if __SSE3__
#include <NEON_2_SSE.h>
#include "sse_mathfun.h"
#endif // __SSE3__

namespace ncnn {

DEFINE_LAYER_CREATOR(Softmax_x86)

int Softmax_x86::forward(const Mat& bottom_blob, Mat& top_blob) const
{
    if (axis != 0)
        return Softmax::forward(bottom_blob, top_blob);

    // value = exp( value - global max value )
    // sum all value
    // value = value / sum

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int size = w * h;

    top_blob.create(w, h, channels);
    if (top_blob.empty())
        return -100;

    Mat max;
    max.create(w, h);
    if (max.empty())
        return -100;
    max.fill(-FLT_MAX);
    for (int q=0; q<channels; q++)
    {
        const float* ptr = bottom_blob.channel(q);
        float* maxptr = max;

        for (int i=0; i<size; i++)
        {
            maxptr[i] = std::max(maxptr[i], ptr[i]);
        }
    }

    #pragma omp parallel for
    for (int q=0; q<channels; q++)
    {
        const float* ptr = bottom_blob.channel(q);
        float* outptr = top_blob.channel(q);
        float* maxptr = max;

#if __SSE3__
        int nn = size >> 2;
        int remain = size - (nn << 2);
#else
        int remain = size;
#endif // __SSE3__

#if __SSE3__
        for (; nn>0; nn--)
        {
            float32x4_t _p = vld1q_f32(ptr);
            float32x4_t _max = vld1q_f32(maxptr);

            _p = exp_ps(vsubq_f32(_p, _max));

            vst1q_f32(outptr, _p);

            ptr += 4;
            maxptr += 4;
            outptr += 4;
        }
#endif // __SSE3__

        for (; remain>0; remain--)
        {
            *outptr = exp(*ptr - *maxptr);

            ptr++;
            maxptr++;
            outptr++;
        }
    }

    Mat sum;
    sum.create(w, h);
    if (sum.empty())
        return -100;
    sum.fill(0.f);
    for (int q=0; q<channels; q++)
    {
        const float* outptr = top_blob.channel(q);
        float* sumptr = sum;

        for (int i=0; i<size; i++)
        {
            sumptr[i] += outptr[i];
        }
    }

    #pragma omp parallel for
    for (int q=0; q<channels; q++)
    {
        float* outptr = top_blob.channel(q);
        float* sumptr = sum;

#if __SSE3__
        int nn = size >> 2;
        int remain = size - (nn << 2);
#else
        int remain = size;
#endif // __SSE3__

#if __SSE3__
        for (; nn>0; nn--)
        {
            float32x4_t _p = vld1q_f32(outptr);
            float32x4_t _sum = vld1q_f32(sumptr);

            _p = vdivq_f32(_p, _sum);
            vst1q_f32(outptr, _p);

            outptr += 4;
            sumptr += 4;
        }
#endif // __SSE3__

        for (; remain>0; remain--)
        {
            *outptr /= *sumptr;

            outptr++;
            sumptr++;
        }
    }

    return 0;
}

int Softmax_x86::forward_inplace(Mat& bottom_top_blob) const
{
    if (axis != 0)
        return Softmax::forward_inplace(bottom_top_blob);

    // value = exp( value - global max value )
    // sum all value
    // value = value / sum

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    Mat max;
    max.create(w, h);
    if (max.empty())
        return -100;
    max.fill(-FLT_MAX);
    for (int q=0; q<channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);
        float* maxptr = max;

        for (int i=0; i<size; i++)
        {
            maxptr[i] = std::max(maxptr[i], ptr[i]);
        }
    }

    #pragma omp parallel for
    for (int q=0; q<channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);
        float* maxptr = max;

#if __SSE3__
        int nn = size >> 2;
        int remain = size - (nn << 2);
#else
        int remain = size;
#endif // __SSE3__

#if __SSE3__
        for (; nn>0; nn--)
        {
            float32x4_t _p = vld1q_f32(ptr);
            float32x4_t _max = vld1q_f32(maxptr);

            _p = exp_ps(vsubq_f32(_p, _max));

            vst1q_f32(ptr, _p);

            ptr += 4;
            maxptr += 4;
        }
#endif // __SSE3__

        for (; remain>0; remain--)
        {
            *ptr = exp(*ptr - *maxptr);

            ptr++;
            maxptr++;
        }
    }

    Mat sum;
    sum.create(w, h);
    if (sum.empty())
        return -100;
    sum.fill(0.f);
    for (int q=0; q<channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);
        float* sumptr = sum;

#if __SSE3__
        int nn = size >> 2;
        int remain = size - (nn << 2);
#else
        int remain = size;
#endif // __SSE3__

#if __SSE3__
        for (; nn>0; nn--)
        {
            float32x4_t _p = vld1q_f32(ptr);
            float32x4_t _sum = vld1q_f32(sumptr);
            _sum = vaddq_f32(_sum, _p);
            vst1q_f32(sumptr, _sum);

            ptr += 4;
            sumptr += 4;
        }
#endif // __SSE3__

        for (; remain>0; remain--)
        {
            *sumptr += *ptr;

            ptr++;
            sumptr++;
        }
    }

    #pragma omp parallel for
    for (int q=0; q<channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);
        float* sumptr = sum;

#if __SSE3__
        int nn = size >> 2;
        int remain = size - (nn << 2);
#else
        int remain = size;
#endif // __SSE3__

#if __SSE3__
        for (; nn>0; nn--)
        {
            float32x4_t _p = vld1q_f32(ptr);
            float32x4_t _sum = vld1q_f32(sumptr);

            _p = vdivq_f32(_p, _sum);
            vst1q_f32(ptr, _p);

            ptr += 4;
            sumptr += 4;
        }
#endif // __SSE3__

        for (; remain>0; remain--)
        {
            *ptr /= *sumptr;

            ptr++;
            sumptr++;
        }
    }

    return 0;
}

} // namespace ncnn
