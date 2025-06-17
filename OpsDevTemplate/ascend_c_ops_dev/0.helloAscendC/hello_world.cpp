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
extern "C" __global__ __aicore__ void hello_world()
{
    printf("Hello World!!!\n");
}

void hello_world_do(uint32_t blockDim, void *stream)
{
    hello_world<<<blockDim, nullptr, stream>>>();
}