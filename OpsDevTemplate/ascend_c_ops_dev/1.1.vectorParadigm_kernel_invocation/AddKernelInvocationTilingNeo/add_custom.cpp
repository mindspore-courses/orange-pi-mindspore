/**
 * @file add_custom.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "add_custom_tiling.h"
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        /***
         * #define GM_ADDR _gm__ uint8_t* __restrict__宏表示此指针变量驻留在Global Memory中某处地址
         */
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        printf("每个核算多少数据totalLength=%d\n",totalLength);
        printf("AscendC::GetBlockNum()=%d\n", AscendC::GetBlockNum());
        printf("tileNum=%d\n",tileNum);
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM; //每一小块计算多长=切片长度=总长度/切片数/缓存数2
        printf("tileLength=%d\n",this->tileLength);
        xGm.SetGlobalBuffer((__gm__ half *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);// half半精度浮点数
        yGm.SetGlobalBuffer((__gm__ half *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ half *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        printf("BUFFER_NUM=%d\n",BUFFER_NUM);
        printf("tilNum=%d\n",this->tileNum);
        printf("loopCount=%d\n",loopCount);
        //  大部分AICore一次只能计算256B, FP16（半精度浮点数）占用2个字节=16位=2B=16b。因此一次大多数只能算256B / 2B = 128个FP16数据。
        //  np.random.uniform(1, 100, [8, 2048]).astype(np.float16)， 如果2048个数每次128，需要2048 / 128 = 16次计算,
        // 0~7号核每个核计算2048个数，BLOCK_LENGTH=128前面计算的长度，tileNum=16切片数，loopCount=16*2=32循环次数。动态shape编译器长度未知，视频处理CV之类的是静态shape编译器长度已知。
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();//local memory申请内存
        AscendC::LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        AscendC::LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        AscendC::LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        AscendC::Add(zLocal, xLocal, yLocal, this->tileLength);
        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        if(0==AscendC::GetBlockIdx()) {
            printf("CopyOut progress=%d\n",progress);
            // AscendC:DumpTensor(zLocal, 10101, tileLength); //打印输出 https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/apiref/ascendcopapi/atlasascendc_api_07_0182.html
        }
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY; // 创建输入队列资源
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ; // 创建输出队列资源
    AscendC::GlobalTensor<half> xGm;
        
    AscendC::GlobalTensor<half> yGm;
    AscendC::GlobalTensor<half> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, AddCustomTilingData tiling)
{
    KernelAdd op;
    op.Init(x, y, z, tiling.totalLength, tiling.tileNum);
    op.Process();
}
