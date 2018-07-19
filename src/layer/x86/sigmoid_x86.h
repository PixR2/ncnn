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

#ifndef LAYER_SIGMOID_x86_H
#define LAYER_SIGMOID_x86_H

#include "sigmoid.h"

namespace ncnn {

class Sigmoid_x86 : public Sigmoid
{
public:
    virtual int forward(const Mat& bottom_blob, Mat& top_blob) const;

    virtual int forward_inplace(Mat& bottom_top_blob) const;
};

} // namespace ncnn

#endif // LAYER_SIGMOID_x86_H