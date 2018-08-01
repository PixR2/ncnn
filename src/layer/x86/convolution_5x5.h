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

static void conv5x5s1_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias)
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

        for (int q=0; q<inch; q++)
        {
            float* outptr = out;
            float* outptr2 = outptr + outw;

            const float* img0 = bottom_blob.channel(q);

            const float* kernel0 = kernel + p*inch*25  + q*25;

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w*2;
            const float* r3 = img0 + w*3;
            const float* r4 = img0 + w*4;
            const float* r5 = img0 + w*5;

            const float* k0 = kernel0;
            const float* k1 = kernel0 + 5;
            const float* k2 = kernel0 + 10;
            const float* k3 = kernel0 + 15;
            const float* k4 = kernel0 + 20;

#if __SSE3__
            float32x4_t _k0123 = vld1q_f32(kernel0);
            float32x4_t _k4567 = vld1q_f32(kernel0+4);
            float32x4_t _k891011 = vld1q_f32(kernel0+8);
            float32x4_t _k12131415 = vld1q_f32(kernel0+12);
            float32x4_t _k16171819 = vld1q_f32(kernel0+16);
            float32x4_t _k20212223 = vld1q_f32(kernel0+20);
            float32x4_t _k24242424 = vdupq_n_f32(kernel0[24]);
#endif // __SSE3__

            int i = 0;

            for (; i+1 < outh; i+=2)
            {

#if __SSE3__
                int nn = outw >> 2;
                int remain = outw - (nn << 2);
#else
                int remain = outw;
#endif // __SSE3__

#if __SSE3__
                for (; nn>0; nn--)
                {
                    float32x4_t _sum = vld1q_f32(outptr);
                    float32x4_t _sum2 = vld1q_f32(outptr2);

                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r04 = vld1q_f32(r0 + 4);
                    float32x4_t _r01 = vextq_f32(_r00, _r04, 1);
                    float32x4_t _r02 = vextq_f32(_r00, _r04, 2);
                    float32x4_t _r03 = vextq_f32(_r00, _r04, 3);

                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r14 = vld1q_f32(r1 + 4);
                    float32x4_t _r11 = vextq_f32(_r10, _r14, 1);
                    float32x4_t _r12 = vextq_f32(_r10, _r14, 2);
                    float32x4_t _r13 = vextq_f32(_r10, _r14, 3);

                    float32x4_t _r20 = vld1q_f32(r2);
                    float32x4_t _r24 = vld1q_f32(r2 + 4);
                    float32x4_t _r21 = vextq_f32(_r20, _r24, 1);
                    float32x4_t _r22 = vextq_f32(_r20, _r24, 2);
                    float32x4_t _r23 = vextq_f32(_r20, _r24, 3);

                    float32x4_t _r30 = vld1q_f32(r3);
                    float32x4_t _r34 = vld1q_f32(r3 + 4);
                    float32x4_t _r31 = vextq_f32(_r30, _r34, 1);
                    float32x4_t _r32 = vextq_f32(_r30, _r34, 2);
                    float32x4_t _r33 = vextq_f32(_r30, _r34, 3);

                    float32x4_t _r40 = vld1q_f32(r4);
                    float32x4_t _r44 = vld1q_f32(r4 + 4);
                    float32x4_t _r41 = vextq_f32(_r40, _r44, 1);
                    float32x4_t _r42 = vextq_f32(_r40, _r44, 2);
                    float32x4_t _r43 = vextq_f32(_r40, _r44, 3);

                    float32x4_t _r50 = vld1q_f32(r5);
                    float32x4_t _r54 = vld1q_f32(r5 + 4);
                    float32x4_t _r51 = vextq_f32(_r50, _r54, 1);
                    float32x4_t _r52 = vextq_f32(_r50, _r54, 2);
                    float32x4_t _r53 = vextq_f32(_r50, _r54, 3);

                    _sum = vfmaq_laneq_f32(_sum, _r00, _k0123, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r01, _k0123, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r02, _k0123, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r03, _k0123, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r04, _k4567, 0);

                    _sum = vfmaq_laneq_f32(_sum, _r10, _k4567, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r11, _k4567, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r12, _k4567, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r13, _k891011, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r14, _k891011, 1);

                    _sum = vfmaq_laneq_f32(_sum, _r20, _k891011, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r21, _k891011, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r22, _k12131415, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r23, _k12131415, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r24, _k12131415, 2);

                    _sum = vfmaq_laneq_f32(_sum, _r30, _k12131415, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r31, _k16171819, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r32, _k16171819, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r33, _k16171819, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r34, _k16171819, 3);

                    _sum = vfmaq_laneq_f32(_sum, _r40, _k20212223, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r41, _k20212223, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r42, _k20212223, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r43, _k20212223, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r44, _k24242424, 0);

                    _sum2 = vfmaq_laneq_f32(_sum2, _r10, _k0123, 0);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r11, _k0123, 1);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r12, _k0123, 2);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r13, _k0123, 3);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r14, _k4567, 0);

                    _sum2 = vfmaq_laneq_f32(_sum2, _r20, _k4567, 1);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r21, _k4567, 2);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r22, _k4567, 3);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r23, _k891011, 0);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r24, _k891011, 1);

                    _sum2 = vfmaq_laneq_f32(_sum2, _r30, _k891011, 2);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r31, _k891011, 3);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r32, _k12131415, 0);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r33, _k12131415, 1);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r34, _k12131415, 2);

                    _sum2 = vfmaq_laneq_f32(_sum2, _r40, _k12131415, 3);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r41, _k16171819, 0);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r42, _k16171819, 1);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r43, _k16171819, 2);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r44, _k16171819, 3);

                    _sum2 = vfmaq_laneq_f32(_sum2, _r50, _k20212223, 0);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r51, _k20212223, 1);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r52, _k20212223, 2);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r53, _k20212223, 3);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r54, _k24242424, 0);

                    vst1q_f32(outptr, _sum);
                    vst1q_f32(outptr2, _sum2);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    r4 += 4;
                    r5 += 4;
                    outptr += 4;
                    outptr2 += 4;
                }
#endif // __SSE3__
                for (; remain>0; remain--)
                {
                    float sum = 0;
                    float sum2 = 0;
#if __SSE3__
                    float32x4_t _r1 = vld1q_f32(r1);
                    float32x4_t _k1 = vld1q_f32(k1);
                    float32x4_t _sum = vmulq_f32(_r1, _k1);
                    float32x4_t _sum2 = vmulq_f32(_r1, _k0123);

                    float32x4_t _r2 = vld1q_f32(r2);
                    float32x4_t _k2 = vld1q_f32(k2);
                    _sum = vmlaq_f32(_sum, _r2, _k2);
                    _sum2 = vmlaq_f32(_sum2, _r2, _k1);

                    float32x4_t _r3 = vld1q_f32(r3);
                    float32x4_t _k3 = vld1q_f32(k3);
                    _sum = vmlaq_f32(_sum, _r3, _k3);
                    _sum2 = vmlaq_f32(_sum2, _r3, _k2);

                    float32x4_t _r4 = vld1q_f32(r4);
                    _sum = vmlaq_f32(_sum, _r4, _k20212223);
                    _sum2 = vmlaq_f32(_sum2, _r4, _k3);

                    float32x4_t _r0 = vld1q_f32(r0);
                    _sum = vmlaq_f32(_sum, _r0, _k0123);
                    float32x4_t _r5 = vld1q_f32(r5);
                    _sum2 = vmlaq_f32(_sum2, _r5, _k20212223);

                    float32x4_t _k_t4;
                    _k_t4 = vsetq_lane_f32(k0[4], _k_t4, 0);
                    _k_t4 = vsetq_lane_f32(k1[4], _k_t4, 1);
                    _k_t4 = vsetq_lane_f32(k2[4], _k_t4, 2);
                    _k_t4 = vsetq_lane_f32(k3[4], _k_t4, 3);

                    float32x4_t _r_t4;

                    _r_t4 = vsetq_lane_f32(r0[4], _r_t4, 0);
                    _r_t4 = vsetq_lane_f32(r1[4], _r_t4, 1);
                    _r_t4 = vsetq_lane_f32(r2[4], _r_t4, 2);
                    _r_t4 = vsetq_lane_f32(r3[4], _r_t4, 3);
                    _sum = vmlaq_f32(_sum, _r_t4, _k_t4);

                    sum = r4[4] * k4[4];

                    _r_t4 = vextq_f32(_r_t4, _r_t4, 1);
                    _r_t4 = vsetq_lane_f32(r4[4], _r_t4, 3);
                    _sum2 = vmlaq_f32(_sum2, _r_t4, _k_t4);

                    sum2 = r5[4] * k4[4];

                    float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                    float32x2_t _ss2 = vadd_f32(vget_low_f32(_sum2), vget_high_f32(_sum2));
                    float32x2_t _ss_ss2 = vpadd_f32(_ss, _ss2);

                    sum += vget_lane_f32(_ss_ss2, 0);
                    sum2 += vget_lane_f32(_ss_ss2, 1);
#else
                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r0[3] * k0[3];
                    sum += r0[4] * k0[4];

                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r1[3] * k1[3];
                    sum += r1[4] * k1[4];

                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];
                    sum += r2[3] * k2[3];
                    sum += r2[4] * k2[4];

                    sum += r3[0] * k3[0];
                    sum += r3[1] * k3[1];
                    sum += r3[2] * k3[2];
                    sum += r3[3] * k3[3];
                    sum += r3[4] * k3[4];

                    sum += r4[0] * k4[0];
                    sum += r4[1] * k4[1];
                    sum += r4[2] * k4[2];
                    sum += r4[3] * k4[3];
                    sum += r4[4] * k4[4];

                    sum2 += r1[0] * k0[0];
                    sum2 += r1[1] * k0[1];
                    sum2 += r1[2] * k0[2];
                    sum2 += r1[3] * k0[3];
                    sum2 += r1[4] * k0[4];

                    sum2 += r2[0] * k1[0];
                    sum2 += r2[1] * k1[1];
                    sum2 += r2[2] * k1[2];
                    sum2 += r2[3] * k1[3];
                    sum2 += r2[4] * k1[4];

                    sum2 += r3[0] * k2[0];
                    sum2 += r3[1] * k2[1];
                    sum2 += r3[2] * k2[2];
                    sum2 += r3[3] * k2[3];
                    sum2 += r3[4] * k2[4];

                    sum2 += r4[0] * k3[0];
                    sum2 += r4[1] * k3[1];
                    sum2 += r4[2] * k3[2];
                    sum2 += r4[3] * k3[3];
                    sum2 += r4[4] * k3[4];

                    sum2 += r5[0] * k4[0];
                    sum2 += r5[1] * k4[1];
                    sum2 += r5[2] * k4[2];
                    sum2 += r5[3] * k4[3];
                    sum2 += r5[4] * k4[4];
#endif // __SSE3__
                    *outptr += sum;
                    *outptr2 += sum2;

                    r0++;
                    r1++;
                    r2++;
                    r3++;
                    r4++;
                    r5++;
                    outptr++;
                    outptr2++;
                }

                r0 += 4 + w;
                r1 += 4 + w;
                r2 += 4 + w;
                r3 += 4 + w;
                r4 += 4 + w;
                r5 += 4 + w;

                outptr += outw;
                outptr2 += outw;
            }

            for (; i < outh; i++)
            {

#if __SSE3__
                int nn = outw >> 2;
                int remain = outw - (nn << 2);
#else
                int remain = outw;
#endif // __SSE3__

#if __SSE3__
                for (; nn>0; nn--)
                {
                    float32x4_t _sum = vld1q_f32(outptr);

                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r04 = vld1q_f32(r0 + 4);
                    float32x4_t _r01 = vextq_f32(_r00, _r04, 1);
                    float32x4_t _r02 = vextq_f32(_r00, _r04, 2);
                    float32x4_t _r03 = vextq_f32(_r00, _r04, 3);

                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r14 = vld1q_f32(r1 + 4);
                    float32x4_t _r11 = vextq_f32(_r10, _r14, 1);
                    float32x4_t _r12 = vextq_f32(_r10, _r14, 2);
                    float32x4_t _r13 = vextq_f32(_r10, _r14, 3);

                    float32x4_t _r20 = vld1q_f32(r2);
                    float32x4_t _r24 = vld1q_f32(r2 + 4);
                    float32x4_t _r21 = vextq_f32(_r20, _r24, 1);
                    float32x4_t _r22 = vextq_f32(_r20, _r24, 2);
                    float32x4_t _r23 = vextq_f32(_r20, _r24, 3);

                    float32x4_t _r30 = vld1q_f32(r3);
                    float32x4_t _r34 = vld1q_f32(r3 + 4);
                    float32x4_t _r31 = vextq_f32(_r30, _r34, 1);
                    float32x4_t _r32 = vextq_f32(_r30, _r34, 2);
                    float32x4_t _r33 = vextq_f32(_r30, _r34, 3);

                    float32x4_t _r40 = vld1q_f32(r4);
                    float32x4_t _r44 = vld1q_f32(r4 + 4);
                    float32x4_t _r41 = vextq_f32(_r40, _r44, 1);
                    float32x4_t _r42 = vextq_f32(_r40, _r44, 2);
                    float32x4_t _r43 = vextq_f32(_r40, _r44, 3);

                    _sum = vfmaq_laneq_f32(_sum, _r00, _k0123, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r01, _k0123, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r02, _k0123, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r03, _k0123, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r04, _k4567, 0);

                    _sum = vfmaq_laneq_f32(_sum, _r10, _k4567, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r11, _k4567, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r12, _k4567, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r13, _k891011, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r14, _k891011, 1);

                    _sum = vfmaq_laneq_f32(_sum, _r20, _k891011, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r21, _k891011, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r22, _k12131415, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r23, _k12131415, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r24, _k12131415, 2);

                    _sum = vfmaq_laneq_f32(_sum, _r30, _k12131415, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r31, _k16171819, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r32, _k16171819, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r33, _k16171819, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r34, _k16171819, 3);

                    _sum = vfmaq_laneq_f32(_sum, _r40, _k20212223, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r41, _k20212223, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r42, _k20212223, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r43, _k20212223, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r44, _k24242424, 0);

                    vst1q_f32(outptr, _sum);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    r4 += 4;
                    outptr += 4;
                }
#endif // __SSE3__
                for (; remain>0; remain--)
                {
                    float sum = 0;
#if __SSE3__
                    float32x4_t _r0 = vld1q_f32(r0);
                    float32x4_t _sum = vmulq_f32(_r0, _k0123);

                    float32x4_t _r1 = vld1q_f32(r1);
                    _sum = vmlaq_f32(_sum, _r1, vld1q_f32(k1));

                    float32x4_t _r2 = vld1q_f32(r2);
                    _sum = vmlaq_f32(_sum, _r2, vld1q_f32(k2));

                    float32x4_t _r3 = vld1q_f32(r3);
                    _sum = vmlaq_f32(_sum, _r3, vld1q_f32(k3));

                    float32x4_t _r4 = vld1q_f32(r4);
                    _sum = vmlaq_f32(_sum, _r4, _k20212223);

                    float32x4_t _k_t4;
                    _k_t4 = vsetq_lane_f32(k0[4], _k_t4, 0);
                    _k_t4 = vsetq_lane_f32(k1[4], _k_t4, 1);
                    _k_t4 = vsetq_lane_f32(k2[4], _k_t4, 2);
                    _k_t4 = vsetq_lane_f32(k3[4], _k_t4, 3);

                    float32x4_t _r_t4;

                    _r_t4 = vsetq_lane_f32(r0[4], _r_t4, 0);
                    _r_t4 = vsetq_lane_f32(r1[4], _r_t4, 1);
                    _r_t4 = vsetq_lane_f32(r2[4], _r_t4, 2);
                    _r_t4 = vsetq_lane_f32(r3[4], _r_t4, 3);
                    _sum = vmlaq_f32(_sum, _r_t4, _k_t4);

                    sum = r4[4] * k4[4];

                    float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                    _ss = vpadd_f32(_ss, _ss);

                    sum += vget_lane_f32(_ss, 0);
#else
                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r0[3] * k0[3];
                    sum += r0[4] * k0[4];

                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r1[3] * k1[3];
                    sum += r1[4] * k1[4];

                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];
                    sum += r2[3] * k2[3];
                    sum += r2[4] * k2[4];

                    sum += r3[0] * k3[0];
                    sum += r3[1] * k3[1];
                    sum += r3[2] * k3[2];
                    sum += r3[3] * k3[3];
                    sum += r3[4] * k3[4];

                    sum += r4[0] * k4[0];
                    sum += r4[1] * k4[1];
                    sum += r4[2] * k4[2];
                    sum += r4[3] * k4[3];
                    sum += r4[4] * k4[4];
#endif
                    *outptr += sum;

                    r0++;
                    r1++;
                    r2++;
                    r3++;
                    r4++;
                    outptr++;
                }

                r0 += 4;
                r1 += 4;
                r2 += 4;
                r3 += 4;
                r4 += 4;

            }

        }
    }

}

static void conv5x5s2_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias)
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

        for (int q=0; q<inch; q++)
        {
            float* outptr = out;

            const float* img0 = bottom_blob.channel(q);

            const float* kernel0 = kernel + p*inch*25  + q*25;

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w*2;
            const float* r3 = img0 + w*3;
            const float* r4 = img0 + w*4;

            const float* k0 = kernel0;
            const float* k1 = kernel0 + 5;
            const float* k2 = kernel0 + 10;
            const float* k3 = kernel0 + 15;
            const float* k4 = kernel0 + 20;

#if __SSE3__
            float32x4_t _k0123 = vld1q_f32(kernel0);
            float32x4_t _k4567 = vld1q_f32(kernel0+4);
            float32x4_t _k891011 = vld1q_f32(kernel0+8);
            float32x4_t _k12131415 = vld1q_f32(kernel0+12);
            float32x4_t _k16171819 = vld1q_f32(kernel0+16);
            float32x4_t _k20212223 = vld1q_f32(kernel0+20);
            float32x4_t _k24242424 = vdupq_n_f32(kernel0[24]);
#endif // __SSE3__

            for (int i = 0; i < outh; i++)
            {

#if __SSE3__
                int nn = outw >> 2;
                int remain = outw - (nn << 2);
#else
                int remain = outw;
#endif // __SSE3__

#if __SSE3__
                for (; nn>0; nn--)
                {
                    float32x4_t _sum = vld1q_f32(outptr);

                    float32x4x2_t _r00_02461357 = vld2q_f32(r0);
                    float32x4x2_t _r00nx2 = vld2q_f32(r0 + 8);
                    float32x4_t _r0_8101214 = _r00nx2.val[0];// 8 10 12 14
                    float32x4_t _r0_9111315 = _r00nx2.val[1];// 9 11 13 15
                    float32x4_t _r00 = _r00_02461357.val[0];// 0 2 4 6
                    float32x4_t _r01 = _r00_02461357.val[1];// 1 3 5 7
                    float32x4_t _r02 = vextq_f32(_r00, _r0_8101214, 1);// 2 4 6 8
                    float32x4_t _r03 = vextq_f32(_r01, _r0_9111315, 1);// 3 5 7 9
                    float32x4_t _r04 = vextq_f32(_r00, _r0_8101214, 2);// 4 6 8 10

                    float32x4x2_t _r10_02461357 = vld2q_f32(r1);
                    float32x4x2_t _r10nx2 = vld2q_f32(r1 + 8);
                    float32x4_t _r1_8101214 = _r10nx2.val[0];
                    float32x4_t _r1_9111315 = _r10nx2.val[1];
                    float32x4_t _r10 = _r10_02461357.val[0];
                    float32x4_t _r11 = _r10_02461357.val[1];
                    float32x4_t _r12 = vextq_f32(_r10, _r1_8101214, 1);
                    float32x4_t _r13 = vextq_f32(_r11, _r1_9111315, 1);
                    float32x4_t _r14 = vextq_f32(_r10, _r1_8101214, 2);

                    float32x4x2_t _r20_02461357 = vld2q_f32(r2);
                    float32x4x2_t _r20nx2 = vld2q_f32(r2 + 8);
                    float32x4_t _r2_8101214 = _r20nx2.val[0];
                    float32x4_t _r2_9111315 = _r20nx2.val[1];
                    float32x4_t _r20 = _r20_02461357.val[0];
                    float32x4_t _r21 = _r20_02461357.val[1];
                    float32x4_t _r22 = vextq_f32(_r20, _r2_8101214, 1);
                    float32x4_t _r23 = vextq_f32(_r21, _r2_9111315, 1);
                    float32x4_t _r24 = vextq_f32(_r20, _r2_8101214, 2);

                    float32x4x2_t _r30_02461357 = vld2q_f32(r3);
                    float32x4x2_t _r30nx2 = vld2q_f32(r3 + 8);
                    float32x4_t _r3_8101214 = _r30nx2.val[0];
                    float32x4_t _r3_9111315 = _r30nx2.val[1];
                    float32x4_t _r30 = _r30_02461357.val[0];
                    float32x4_t _r31 = _r30_02461357.val[1];
                    float32x4_t _r32 = vextq_f32(_r30, _r3_8101214, 1);
                    float32x4_t _r33 = vextq_f32(_r31, _r3_9111315, 1);
                    float32x4_t _r34 = vextq_f32(_r30, _r3_8101214, 2);

                    float32x4x2_t _r40_02461357 = vld2q_f32(r4);
                    float32x4x2_t _r40nx2 = vld2q_f32(r4 + 8);
                    float32x4_t _r4_8101214 = _r40nx2.val[0];
                    float32x4_t _r4_9111315 = _r40nx2.val[1];
                    float32x4_t _r40 = _r40_02461357.val[0];
                    float32x4_t _r41 = _r40_02461357.val[1];
                    float32x4_t _r42 = vextq_f32(_r40, _r4_8101214, 1);
                    float32x4_t _r43 = vextq_f32(_r41, _r4_9111315, 1);
                    float32x4_t _r44 = vextq_f32(_r40, _r4_8101214, 2);

                    _sum = vfmaq_laneq_f32(_sum, _r00, _k0123, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r01, _k0123, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r02, _k0123, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r03, _k0123, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r04, _k4567, 0);

                    _sum = vfmaq_laneq_f32(_sum, _r10, _k4567, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r11, _k4567, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r12, _k4567, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r13, _k891011, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r14, _k891011, 1);

                    _sum = vfmaq_laneq_f32(_sum, _r20, _k891011, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r21, _k891011, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r22, _k12131415, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r23, _k12131415, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r24, _k12131415, 2);

                    _sum = vfmaq_laneq_f32(_sum, _r30, _k12131415, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r31, _k16171819, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r32, _k16171819, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r33, _k16171819, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r34, _k16171819, 3);

                    _sum = vfmaq_laneq_f32(_sum, _r40, _k20212223, 0);
                    _sum = vfmaq_laneq_f32(_sum, _r41, _k20212223, 1);
                    _sum = vfmaq_laneq_f32(_sum, _r42, _k20212223, 2);
                    _sum = vfmaq_laneq_f32(_sum, _r43, _k20212223, 3);
                    _sum = vfmaq_laneq_f32(_sum, _r44, _k24242424, 0);

                    vst1q_f32(outptr, _sum);

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                    r3 += 8;
                    r4 += 8;
                    outptr += 4;
                }
#endif // __SSE3__
                for (; remain>0; remain--)
                {
                    float sum = 0;
#if __SSE3__
                    float32x4_t _r0 = vld1q_f32(r0);
                    float32x4_t _sum = vmulq_f32(_r0, _k0123);

                    float32x4_t _r1 = vld1q_f32(r1);
                    _sum = vmlaq_f32(_sum, _r1, vld1q_f32(k1));

                    float32x4_t _r2 = vld1q_f32(r2);
                    _sum = vmlaq_f32(_sum, _r2, vld1q_f32(k2));

                    float32x4_t _r3 = vld1q_f32(r3);
                    _sum = vmlaq_f32(_sum, _r3, vld1q_f32(k3));

                    float32x4_t _r4 = vld1q_f32(r4);
                    _sum = vmlaq_f32(_sum, _r4, _k20212223);

                    sum += r0[4] * k0[4];
                    sum += r1[4] * k1[4];
                    sum += r2[4] * k2[4];
                    sum += r3[4] * k3[4];
                    sum += r4[4] * k4[4];

                    float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                    _ss = vpadd_f32(_ss, _ss);

                    sum += vget_lane_f32(_ss, 0);
#else
                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r0[3] * k0[3];
                    sum += r0[4] * k0[4];

                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r1[3] * k1[3];
                    sum += r1[4] * k1[4];

                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];
                    sum += r2[3] * k2[3];
                    sum += r2[4] * k2[4];

                    sum += r3[0] * k3[0];
                    sum += r3[1] * k3[1];
                    sum += r3[2] * k3[2];
                    sum += r3[3] * k3[3];
                    sum += r3[4] * k3[4];

                    sum += r4[0] * k4[0];
                    sum += r4[1] * k4[1];
                    sum += r4[2] * k4[2];
                    sum += r4[3] * k4[3];
                    sum += r4[4] * k4[4];
#endif
                    *outptr += sum;

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    r3 += 2;
                    r4 += 2;
                    outptr++;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
                r3 += tailstep;
                r4 += tailstep;
            }

        }
    }

}
