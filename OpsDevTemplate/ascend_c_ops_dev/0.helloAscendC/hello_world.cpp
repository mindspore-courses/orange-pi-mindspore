/**
 * @file hello_world.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"
using namespace AscendC;
// __global__ 声明这些代码运行在Ascend NPU上，必须是void返回值类型，可以被<<<>>>调用
// __aicore__ 此核函数在Ascend NPU上AICore运行
extern "C" __global__ __aicore__ void hello_world()
{
    printf("Hello World!!!\n");
}

void hello_world_do(uint32_t blockDim, void *stream)
{
    hello_world<<<blockDim, nullptr, stream>>>();
    // blockDim, nullptr, stream 参数，blockDim表示每个卡号，stream表示流
}