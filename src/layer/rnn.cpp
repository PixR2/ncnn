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

#include "rnn.h"
#include <cmath>

namespace ncnn {

DEFINE_LAYER_CREATOR(RNN)

RNN::RNN()
{
    one_blob_only = false;
    support_inplace = false;
}

RNN::~RNN()
{
}

int RNN::load_param(const ParamDict& pd)
{
    num_output = pd.get(0, 0);
    weight_data_size = pd.get(1, 0);

    return 0;
}

#if NCNN_STDIO
int RNN::load_model(FILE* binfp)
{
    int nread;

    int size = (weight_data_size - num_output * num_output) / 2 / num_output;

    // raw weight data
    weight_hh_data.create(size, num_output);
    if (weight_hh_data.empty())
        return -100;
    nread = fread(weight_hh_data.data, size * num_output * sizeof(float), 1, binfp);
    if (nread != 1)
    {
        fprintf(stderr, "RNN read weight_hh_data failed %d\n", nread);
        return -1;
    }

    weight_xh_data.create(size, num_output);
    if (weight_xh_data.empty())
        return -100;
    nread = fread(weight_xh_data.data, size * num_output * sizeof(float), 1, binfp);
    if (nread != 1)
    {
        fprintf(stderr, "RNN read weight_xh_data failed %d\n", nread);
        return -1;
    }

    weight_ho_data.create(num_output, num_output);
    if (weight_ho_data.empty())
        return -100;
    nread = fread(weight_ho_data.data, num_output * num_output * sizeof(float), 1, binfp);
    if (nread != 1)
    {
        fprintf(stderr, "RNN read weight_ho_data failed %d\n", nread);
        return -1;
    }

    bias_h_data.create(num_output);
    if (bias_h_data.empty())
        return -100;
    nread = fread(bias_h_data.data, num_output * sizeof(float), 1, binfp);
    if (nread != 1)
    {
        fprintf(stderr, "RNN read bias_h_data failed %d\n", nread);
        return -1;
    }

    bias_o_data.create(num_output);
    if (bias_o_data.empty())
        return -100;
    nread = fread(bias_o_data.data, num_output * sizeof(float), 1, binfp);
    if (nread != 1)
    {
        fprintf(stderr, "RNN read bias_o_data failed %d\n", nread);
        return -1;
    }

    return 0;
}
#endif // NCNN_STDIO

int RNN::load_model(const unsigned char*& mem)
{
    int size = (weight_data_size - num_output * num_output) / 2 / num_output;

    // raw weight data
    weight_hh_data = Mat(size, num_output, (float*)mem);
    mem += size * num_output * sizeof(float);

    weight_xh_data = Mat(size, num_output, (float*)mem);
    mem += size * num_output * sizeof(float);

    weight_ho_data = Mat(num_output, num_output, (float*)mem);
    mem += num_output * num_output * sizeof(float);

    bias_h_data = Mat(num_output, (float*)mem);
    mem += num_output * sizeof(float);

    bias_o_data = Mat(num_output, (float*)mem);
    mem += num_output * sizeof(float);

    return 0;
}

int RNN::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const
{
    // size x 1 x T
    const Mat& input_blob = bottom_blobs[0];

    // T, 0 or 1 each
    const Mat& cont_blob = bottom_blobs[1];

    int T = input_blob.c;
    int size = input_blob.w;

    // initial hidden state
    Mat hidden(num_output);
    if (hidden.empty())
        return -100;
    hidden.fill(0.f);

    Mat& top_blob = top_blobs[0];
    top_blob.create(num_output, 1, T);
    if (top_blob.empty())
        return -100;

    // unroll
    for (int t=0; t<T; t++)
    {
        // clip hidden by continuation indicator
        // h_cont_{t-1} = cont_t * h_{t-1}
        // h_cont_{t-1} = h_{t-1} if cont_t == 1
        //                0       otherwise
        // calculate hidden
        // h_t = tanh( W_hh * h_cont_{t-1} + W_xh * x_t + b_h )
        const float cont = cont_blob.data[t];
        const Mat x = input_blob.channel(t);
        float* hidden_data = hidden;
        for (int q=0; q<num_output; q++)
        {
            float h_cont = cont ? hidden_data[q] : 0.f;

            const float* weight_hh_data_ptr = weight_hh_data.data + weight_hh_data.w * q;
            const float* weight_xh_data_ptr = weight_xh_data.data + weight_xh_data.w * q;
            const float* x_data = x;

            float s0 = bias_h_data.data[q];
            for (int i=0; i<size; i++)
            {
                s0 += weight_hh_data_ptr[i] * h_cont + weight_xh_data_ptr[i] * x_data[i];
            }

            hidden_data[q] = tanh(s0);
        }

        // calculate output
        // o_t = tanh( W_ho * h_t + b_o )
        Mat output = top_blob.channel(t);
        float* output_data = output;
        for (int q=0; q<num_output; q++)
        {
            const float* weight_ho_data_ptr = weight_ho_data.data + weight_ho_data.w * q;

            float s0 = bias_o_data.data[q];
            for (int i=0; i<size; i++)
            {
                s0 += weight_ho_data_ptr[i] * hidden_data[i];
            }

            output_data[q] = tanh(s0);
        }

        // no hidden output here
    }

    return 0;
}

} // namespace ncnn
