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

#include "slice_x86.h"
#if __SSE3__
#include "NEON_2_SSE.h"
#endif // __SSE3__

namespace ncnn {

DEFINE_LAYER_CREATOR(Slice_x86)

int Slice_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;

    int q = 0;
    const int* slices_ptr = (const int*)slices.data;
    for (size_t i=0; i<top_blobs.size(); i++)
    {
        int slice = slices_ptr[i];
        if (slice == -233)
        {
            slice = (channels - q) / (top_blobs.size() - i);
        }

        Mat& top_blob = top_blobs[i];
        top_blob.create(w, h, slice);
        if (top_blob.empty())
            return -100;

        int size = bottom_blob.cstep * slice;

        const float* ptr = bottom_blob.channel(q);
        float* outptr = top_blob.data;

#if __SSE3__
        int nn = size >> 3;
        int remain = size - (nn << 3);
#else
        int remain = size;
#endif // __SSE3__

#if __SSE3__
        for (; nn>0; nn--)
        {
            float32x4_t _p = vld1q_f32(ptr);
            float32x4_t _p2 = vld1q_f32(ptr+4);
            vst1q_f32(outptr, _p);
            vst1q_f32(outptr+4, _p2);

            ptr += 8;
            outptr += 8;
        }
#endif // __SSE3__
        for (; remain>0; remain--)
        {
            *outptr++ = *ptr++;
        }

        q += slice;
    }

    return 0;
}

} // namespace ncnn
