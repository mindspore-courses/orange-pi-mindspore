/**
 * @file main.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "acl/acl.h"
extern void hello_world_do(uint32_t coreDim, void *stream);

int32_t main(int argc, char const *argv[])
{
/**
 * @brief hello_world for Ascend C
 */
    aclInit(nullptr); // AscendCL初始化
    // 运行管理资源申请
    int32_t deviceId = 0;
    aclrtSetDevice(deviceId);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    // 设置参与运算的核数为8
    constexpr uint32_t blockDim = 8;
    hello_world_do(blockDim, stream);
    aclrtSynchronizeStream(stream);
    // 资源释放和AscendCL去初始化
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
