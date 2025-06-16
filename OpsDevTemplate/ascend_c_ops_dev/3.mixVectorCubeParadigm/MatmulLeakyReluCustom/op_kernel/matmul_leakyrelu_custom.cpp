/**
 * @file matmul_leakyrelu_custom.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

template <typename aType, typename bType, typename cType, typename biasType> class MatmulLeakyKernel {
public:
    __aicore__ inline MatmulLeakyKernel(){};
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace,
                                const TCubeTiling &tiling, float alpha, AscendC::TPipe *pipe);
    template <bool setTmpSpace = false> __aicore__ inline void Process(AscendC::TPipe *pipe);

    __aicore__ inline void MatmulCompute();
    __aicore__ inline void LeakyReluCompute();
    __aicore__ inline void CopyOut(uint32_t count);
    __aicore__ inline void CalcOffset(int32_t blockIdx, const TCubeTiling &tiling, int32_t &offsetA, int32_t &offsetB,
                                      int32_t &offsetC, int32_t &offsetBias);

    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType>, MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bType>,
           MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, cType>, MatmulType<AscendC::TPosition::GM, CubeFormat::ND, biasType>>
        matmulObj;

    AscendC::GlobalTensor<aType> aGlobal;
    AscendC::GlobalTensor<bType> bGlobal;
    AscendC::GlobalTensor<cType> cGlobal;
    AscendC::GlobalTensor<biasType> biasGlobal;
    AscendC::LocalTensor<cType> reluOutLocal;
    float alpha;
    TCubeTiling tiling;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> reluOutQueue_;
};

/**
  * @brief  Set matmulLeaky input and output gm addr of current core.
  * @param  a: A matrix gm addr.
  * @param  b: B matrix gm addr.
  * @param  bias: Bias gm addr.
  * @param  c: C matrix gm addr.
  * @param  workspace: Temporary gm space addr required by matmul calc.
  * @param  tiling: matmul tiling data.
  * @param  alpha: leaky tiling data.
  * @param  pipe: Global memory and sync management TPipe object.
  * @retval None
  */
template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void
MatmulLeakyKernel<aType, bType, cType, biasType>::Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace,
                                                       const TCubeTiling &tiling, float alpha, AscendC::TPipe *pipe)
{
    this->tiling = tiling;
    this->alpha = alpha;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bType *>(b), tiling.Kb * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c), tiling.M * tiling.N);
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ biasType *>(bias), tiling.N);

    int offsetA = 0;
    int offsetB = 0;
    int offsetC = 0;
    int offsetBias = 0;
    CalcOffset(AscendC::GetBlockIdx(), tiling, offsetA, offsetB, offsetC, offsetBias); // Calculate the gm offset based on the blockidx.
    aGlobal = aGlobal[offsetA];
    bGlobal = bGlobal[offsetB];
    cGlobal = cGlobal[offsetC];
    biasGlobal = biasGlobal[offsetBias];
    pipe->InitBuffer(reluOutQueue_, 1, tiling.baseM * tiling.baseN * sizeof(cType)); // Init output buffer.
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
}

/**
  * @brief  Main process of matmul calculation
  * @param  pipe: Global memory and sync management TPipe object.
  * @retval None
  */
template <typename aType, typename bType, typename cType, typename biasType>
template <bool setTmpSpace>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::Process(AscendC::TPipe *pipe)
{
    uint32_t computeRound = 0;
    // Set temp UB space if the setTmpSpace is true.
    if constexpr (setTmpSpace) {
        AscendC::TBuf<> tmpMMFormatUb;
        AscendC::LocalTensor<uint8_t> mmformatUb;
        pipe->InitBuffer(tmpMMFormatUb, tiling.baseM * tiling.baseN * sizeof(cType));
        mmformatUb = tmpMMFormatUb.Get<uint8_t>(tiling.baseM * tiling.baseN * sizeof(cType));
        matmulObj.SetLocalWorkspace(mmformatUb);
    }
    matmulObj.SetTensorA(aGlobal);
    matmulObj.SetTensorB(bGlobal);
    matmulObj.SetBias(biasGlobal);
    while (matmulObj.template Iterate<true>()) { // Once Iterate, compute baseM * baseN, sync is set true here.
        MatmulCompute(); // Get matmul compute result.
        LeakyReluCompute(); // Compute leakyRelu.
        CopyOut(computeRound); // Copy leakyRelu out result to GM.
        computeRound++;
    }
    matmulObj.End();
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::MatmulCompute()
{
    reluOutLocal = reluOutQueue_.AllocTensor<cType>();
    matmulObj.template GetTensorC<true>(reluOutLocal, false, true);
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::LeakyReluCompute()
{
    LeakyRelu(reluOutLocal, reluOutLocal, (cType)alpha, tiling.baseM * tiling.baseN);
    reluOutQueue_.EnQue(reluOutLocal);
}

/**
  * @brief  Copy leakyRelu out result to GM.
  * @param  count: Iterate count(once Iterate, compute baseM * baseN).
  * @retval None
  */
template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::CopyOut(uint32_t count)
{
    reluOutQueue_.DeQue<cType>();
    const uint32_t roundM = tiling.singleCoreM / tiling.baseM;
    const uint32_t roundN = tiling.singleCoreN / tiling.baseN;
    uint32_t startOffset = (count % roundM * tiling.baseM * tiling.N + count / roundM * tiling.baseN);
    DataCopyParams copyParam = {(uint16_t)tiling.baseM, (uint16_t)(tiling.baseN * sizeof(cType) / DEFAULT_C0_SIZE), 0,
                                (uint16_t)((tiling.N - tiling.baseN) * sizeof(cType) / DEFAULT_C0_SIZE)};
    DataCopy(cGlobal[startOffset], reluOutLocal, copyParam);
    reluOutQueue_.FreeTensor(reluOutLocal);
}

/**
  * @brief  Calculate the gm offset based on the blockidx.
  * @param  blockIdx: Current Core blockidx.
  * @param  tiling: Matmul tiling data.
  * @param  offsetA: Gm offset of A matrix.
  * @param  offsetB: Gm offset of B matrix.
  * @param  offsetC: Gm offset of C matrix.
  * @param  offsetBias: Gm offset of Bias matrix.
  * @retval None
  */
template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void
MatmulLeakyKernel<aType, bType, cType, biasType>::CalcOffset(int32_t blockIdx, const TCubeTiling &tiling,
                                                             int32_t &offsetA, int32_t &offsetB, int32_t &offsetC,
                                                             int32_t &offsetBias)
{
    auto mSingleBlocks = Ceiling(tiling.M, tiling.singleCoreM);
    auto mCoreIndx = blockIdx % mSingleBlocks;
    auto nCoreIndx = blockIdx / mSingleBlocks;

    offsetA = mCoreIndx * tiling.Ka * tiling.singleCoreM;
    offsetB = nCoreIndx * tiling.singleCoreN;
    offsetC = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;
    offsetBias = nCoreIndx * tiling.singleCoreN;
}

/**
  * @brief  matmul_leakyrelu kernel function entry
  * @param  a: A matrix gm addr.
  * @param  b: B matrix gm addr.
  * @param  bias: Bias gm addr.
  * @param  c: Out gm addr.
  * @param  workspace: Temporary gm space addr required by matmul calc.
  * @param  tiling: Tiling data addr. 
  * @retval None
  */
extern "C" __global__ __aicore__ void matmul_leakyrelu_custom(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c,
                                                              GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    MatmulLeakyKernel<half, half, float, float> matmulLeakyKernel;
    AscendC::TPipe pipe;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulLeakyKernel.matmulObj, &tilingData.cubeTilingData); // Initialize the matmul object.
    matmulLeakyKernel.Init(a, b, bias, c, workspace, tilingData.cubeTilingData, tilingData.alpha, &pipe);
    if (TILING_KEY_IS(1)) {
        matmulLeakyKernel.Process(&pipe);
    } else if (TILING_KEY_IS(2)) {
        matmulLeakyKernel.Process<true>(&pipe);
    }
}