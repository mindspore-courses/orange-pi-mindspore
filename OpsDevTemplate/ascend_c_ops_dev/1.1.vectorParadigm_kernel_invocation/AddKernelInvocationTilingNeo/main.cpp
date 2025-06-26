/**
 * @file main.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "add_custom_tiling.h"
#include "data_utils.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_add_custom.h"
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, AddCustomTilingData tiling);
#endif
// #define ASCENDC_CPU_DEBUG 1
int32_t main(int32_t argc, char *argv[])
{
    uint32_t blockDim = 8;
    size_t tilingSize = 2 * sizeof(uint32_t); // 数据分片大小，用于AddCustomTilingData结构体数据切分给不同卡计算偏移量取出来值，然后异步计算SIMD
    size_t inputByteSize = 8 * 2048 * sizeof(uint16_t); // x = np.random.uniform(1, 100, [8, 2048]).astype(np.float16)
    size_t outputByteSize = 8 * 2048 * sizeof(uint16_t);
#ifdef ASCENDC_CPU_DEBUG
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    ReadFile("./input/input_tiling.bin", tilingSize, tiling, tilingSize);
    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *z = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    ReadFile("./input/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile("./input/input_y.bin", inputByteSize, y, inputByteSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE); // 矢量算子需要设置内核模式为AIV模式

    ICPU_RUN_KF(add_custom, blockDim, x, y, z,
                *reinterpret_cast<AddCustomTilingData *>(tiling)); // use this macro for cpu debug

    WriteFile("./output/output_z.bin", z, outputByteSize);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)z);
    AscendC::GmFree((void *)tiling);
#else
    // AscendCL 初始化
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    AddCustomTilingData *tiling;
    uint8_t *xHost, *yHost, *zHost;
    uint8_t *xDevice, *yDevice, *zDevice;

    CHECK_ACL(aclrtMallocHost((void **)(&tiling), tilingSize));
    ReadFile("./input/input_tiling.bin", tilingSize, tiling, tilingSize);
    // 分配Host内存
    CHECK_ACL(aclrtMallocHost((void **)(&xHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&yHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&zHost), outputByteSize));
    CHECK_ACL(aclrtMalloc((void **)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&yDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&zDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    // 初始化Host内存数据
    ReadFile("./input/input_x.bin", inputByteSize, xHost, inputByteSize);
    ReadFile("./input/input_y.bin", inputByteSize, yHost, inputByteSize);
    // 分配Deice内存并拷贝数据到Device
    CHECK_ACL(aclrtMemcpy(xDevice, inputByteSize, xHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(yDevice, inputByteSize, yHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    // “内核调用符”三尖括号本质上重载了这个异步函数，add_custom<<<blockDim, nullptr, stream>>>(xDevice, yDevice, zDevice, tiling);扔进去计算任务丢列
    ACLRT_LAUNCH_KERNEL(add_custom)(blockDim, stream, xDevice, yDevice, zDevice,
                                    tiling);
    CHECK_ACL(aclrtSynchronizeStream(stream)); // 核函数的调用是异步的，核函数的调用结束后，控制权立刻返回给主机端，可以调用以下aclrtSynchronizeStream函数来强制主机端程序等待所有任务队列里核函数执行完毕。
    // 拷贝Device结果数据到Host
    CHECK_ACL(aclrtMemcpy(zHost, outputByteSize, zDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output_z.bin", zHost, outputByteSize);
    // 释放申请的资源
    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(yDevice));
    CHECK_ACL(aclrtFree(zDevice));
    CHECK_ACL(aclrtFreeHost(xHost));
    CHECK_ACL(aclrtFreeHost(yHost));
    CHECK_ACL(aclrtFreeHost(zHost));
    CHECK_ACL(aclrtFreeHost(tiling));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    return 0;
}
