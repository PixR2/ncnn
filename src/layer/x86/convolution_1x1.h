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

static void conv1x1s1_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const float* kernel = _kernel;
    const float* bias = _bias;

    #pragma omp parallel for
    for (int p=0; p<outch; p++)
    {
        Mat out = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        out.fill(bias0);

        int q = 0;

        for (; q+3<inch; q+=4)
        {
            float* outptr = out;

            const float* img0 = bottom_blob.channel(q);
            const float* img1 = bottom_blob.channel(q+1);
            const float* img2 = bottom_blob.channel(q+2);
            const float* img3 = bottom_blob.channel(q+3);

            const float* kernel0 = kernel + p*inch  + q;
            const float k0 = kernel0[0];
            const float k1 = kernel0[1];
            const float k2 = kernel0[2];
            const float k3 = kernel0[3];

            const float* r0 = img0;
            const float* r1 = img1;
            const float* r2 = img2;
            const float* r3 = img3;

            int size = outw * outh;

#if __SSE3__
            int nn = size >> 3;
            int remain = size & 7;
#else
            int remain = size;
#endif // __SSE3__

#if __SSE3__
            float32x4_t _k0 = vdupq_n_f32(k0);
            float32x4_t _k1 = vdupq_n_f32(k1);
            float32x4_t _k2 = vdupq_n_f32(k2);
            float32x4_t _k3 = vdupq_n_f32(k3);
            for (; nn>0; nn--)
            {
                float32x4_t _p = vld1q_f32(r0);
                float32x4_t _pn = vld1q_f32(r0+4);

                float32x4_t _outp = vld1q_f32(outptr);
                float32x4_t _outpn = vld1q_f32(outptr+4);

                _outp = vfmaq_f32(_outp, _p, _k0);
                _outpn = vfmaq_f32(_outpn, _pn, _k0);

                float32x4_t _p1 = vld1q_f32(r1);
                float32x4_t _p1n = vld1q_f32(r1+4);

                _outp = vfmaq_f32(_outp, _p1, _k1);
                _outpn = vfmaq_f32(_outpn, _p1n, _k1);

                float32x4_t _p2 = vld1q_f32(r2);
                float32x4_t _p2n = vld1q_f32(r2+4);

                _outp = vfmaq_f32(_outp, _p2, _k2);
                _outpn = vfmaq_f32(_outpn, _p2n, _k2);

                float32x4_t _p3 = vld1q_f32(r3);
                float32x4_t _p3n = vld1q_f32(r3+4);

                _outp = vfmaq_f32(_outp, _p3, _k3);
                _outpn = vfmaq_f32(_outpn, _p3n, _k3);

                vst1q_f32(outptr, _outp);
                vst1q_f32(outptr+4, _outpn);

                r0 += 8;
                r1 += 8;
                r2 += 8;
                r3 += 8;
                outptr += 8;
            }
#endif // __SSE3__
            for (; remain>0; remain--)
            {
                float sum = *r0 * k0;
                float sum1 = *r1 * k1;
                float sum2 = *r2 * k2;
                float sum3 = *r3 * k3;

                *outptr += sum + sum1 + sum2 + sum3;

                r0++;
                r1++;
                r2++;
                r3++;
                outptr++;
            }

        }

        for (; q<inch; q++)
        {
            float* outptr = out;

            const float* img0 = bottom_blob.channel(q);

            const float* kernel0 = kernel + p*inch  + q;
            const float k0 = kernel0[0];

            const float* r0 = img0;

            int size = outw * outh;

#if __SSE3__
            int nn = size >> 3;
            int remain = size & 7;
#else
            int remain = size;
#endif // __SSE3__

#if __SSE3__
            float32x4_t _k0 = vdupq_n_f32(k0);
            for (; nn>0; nn--)
            {
                float32x4_t _p = vld1q_f32(r0);
                float32x4_t _outp = vld1q_f32(outptr);

                float32x4_t _pn = vld1q_f32(r0+4);
                float32x4_t _outpn = vld1q_f32(outptr+4);

                _outp = vfmaq_f32(_outp, _p, _k0);
                _outpn = vfmaq_f32(_outpn, _pn, _k0);

                vst1q_f32(outptr, _outp);
                vst1q_f32(outptr+4, _outpn);

                r0 += 8;
                outptr += 8;
            }
#endif // __SSE3__
            for (; remain>0; remain--)
            {
                float sum = *r0 * k0;

                *outptr += sum;

                r0++;
                outptr++;
            }

        }
    }

}

static void conv1x1s2_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias)
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    int outch = top_blob.c;

    const int tailstep = w - 2*outw + w;

    const float* kernel = _kernel;
    const float* bias = _bias;

    #pragma omp parallel for
    for (int p=0; p<outch; p++)
    {
        Mat out = top_blob.channel(p);

        const float bias0 = bias ? bias[p] : 0.f;

        out.fill(bias0);

        int q = 0;

        for (; q+3<inch; q+=4)
        {
            float* outptr = out;

            const float* img0 = bottom_blob.channel(q);
            const float* img1 = bottom_blob.channel(q+1);
            const float* img2 = bottom_blob.channel(q+2);
            const float* img3 = bottom_blob.channel(q+3);

            const float* kernel0 = kernel + p*inch  + q;
            const float k0 = kernel0[0];
            const float k1 = kernel0[1];
            const float k2 = kernel0[2];
            const float k3 = kernel0[3];

            const float* r0 = img0;
            const float* r1 = img1;
            const float* r2 = img2;
            const float* r3 = img3;

            for (int i = 0; i < outh; i++)
            {
#if __SSE3__
                int nn = outw >> 3;
                int remain = outw & 7;
#else
                int remain = outw;
#endif // __SSE3__

#if __SSE3__
                float32x4_t _k0 = vdupq_n_f32(k0);
                float32x4_t _k1 = vdupq_n_f32(k1);
                float32x4_t _k2 = vdupq_n_f32(k2);
                float32x4_t _k3 = vdupq_n_f32(k3);
                for (; nn>0; nn--)
                {
                    float32x4x2_t _px2 = vld2q_f32(r0);
                    float32x4_t _p = _px2.val[0];
                    float32x4_t _outp = vld1q_f32(outptr);

                    float32x4x2_t _pnx2 = vld2q_f32(r0+8);
                    float32x4_t _pn = _pnx2.val[0];
                    float32x4_t _outpn = vld1q_f32(outptr+4);

                    _outp = vmlaq_f32(_outp, _p, _k0);
                    _outpn = vmlaq_f32(_outpn, _pn, _k0);

                    float32x4x2_t _p1x2 = vld2q_f32(r1);
                    float32x4_t _p1 = _p1x2.val[0];
                    float32x4x2_t _p1nx2 = vld2q_f32(r1+8);
                    float32x4_t _p1n = _p1nx2.val[0];

                    _outp = vmlaq_f32(_outp, _p1, _k1);
                    _outpn = vmlaq_f32(_outpn, _p1n, _k1);

                    float32x4x2_t _p2x2 = vld2q_f32(r2);
                    float32x4_t _p2 = _p2x2.val[0];
                    float32x4x2_t _p2nx2 = vld2q_f32(r2+8);
                    float32x4_t _p2n = _p2nx2.val[0];

                    _outp = vmlaq_f32(_outp, _p2, _k2);
                    _outpn = vmlaq_f32(_outpn, _p2n, _k2);

                    float32x4x2_t _p3x2 = vld2q_f32(r3);
                    float32x4_t _p3 = _p3x2.val[0];
                    float32x4x2_t _p3nx2 = vld2q_f32(r3+8);
                    float32x4_t _p3n = _p3nx2.val[0];

                    _outp = vmlaq_f32(_outp, _p3, _k3);
                    _outpn = vmlaq_f32(_outpn, _p3n, _k3);

                    vst1q_f32(outptr, _outp);
                    vst1q_f32(outptr+4, _outpn);

                    r0 += 16;
                    r1 += 16;
                    r2 += 16;
                    r3 += 16;
                    outptr += 8;
                }
#endif // __SSE3__
                for (; remain>0; remain--)
                {
                    float sum = *r0 * k0;
                    float sum1 = *r1 * k1;
                    float sum2 = *r2 * k2;
                    float sum3 = *r3 * k3;

                    *outptr += sum + sum1 + sum2 + sum3;

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    r3 += 2;
                    outptr++;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
                r3 += tailstep;
            }

        }

        for (; q<inch; q++)
        {
            float* outptr = out;

            const float* img0 = bottom_blob.channel(q);

            const float* kernel0 = kernel + p*inch  + q;
            const float k0 = kernel0[0];

            const float* r0 = img0;

            for (int i = 0; i < outh; i++)
            {
#if __SSE3__
                int nn = outw >> 3;
                int remain = outw & 7;
#else
                int remain = outw;
#endif // __SSE3__

#if __SSE3__
                float32x4_t _k0 = vdupq_n_f32(k0);
                for (; nn>0; nn--)
                {
                    float32x4x2_t _px2 = vld2q_f32(r0);
                    float32x4_t _p = _px2.val[0];
                    float32x4_t _outp = vld1q_f32(outptr);

                    float32x4x2_t _pnx2 = vld2q_f32(r0+8);
                    float32x4_t _pn = _pnx2.val[0];
                    float32x4_t _outpn = vld1q_f32(outptr+4);

                    _outp = vmlaq_f32(_outp, _p, _k0);
                    _outpn = vmlaq_f32(_outpn, _pn, _k0);

                    vst1q_f32(outptr, _outp);
                    vst1q_f32(outptr+4, _outpn);

                    r0 += 16;
                    outptr += 8;
                }
#endif // __SSE3__
                for (; remain>0; remain--)
                {
                    float sum = *r0 * k0;

                    *outptr += sum;

                    r0 += 2;
                    outptr++;
                }

                r0 += tailstep;
            }

        }
    }

}
