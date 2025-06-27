/**
 * @file matmul_custom.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "matmul_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
using namespace matmul_tiling;

namespace optiling {
/**
  * @brief  Generate matmul tiling.
  * @param  context: Tiling kernel context.
  * @retval Status of GetTiling (GRAPH_SUCCESS or GRAPH_FAILED).
  */
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto shape_a = context->GetInputTensor(0)->GetOriginShape();
    auto shape_b = context->GetInputTensor(1)->GetOriginShape();
    int32_t M = shape_a.GetDim(0);
    int32_t N = shape_b.GetDim(1);
    int32_t K = shape_a.GetDim(1);
    int32_t baseM = 128;
    int32_t baseN = 128;
    MultiCoreMatmulTiling cubeTiling(ascendcPlatform);
    cubeTiling.SetDim(2); // Set the number of cores that participate in multi-core computaion is 2.
    cubeTiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    cubeTiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    cubeTiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetShape(M, N, K);
    cubeTiling.SetOrgShape(M, N, K);
    cubeTiling.SetFixSplit(baseM, baseN, -1); // Set the fixed baseM=128, baseN=128.
    cubeTiling.SetBias(true);
    cubeTiling.SetBufferSpace(-1, -1, -1);
    MatmulCustomTilingData tiling;
    if (cubeTiling.GetTiling(tiling.cubeTilingData) == -1) {
        return ge::GRAPH_FAILED;
    }
    uint64_t localMemSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, localMemSize);
    tiling.set_localMemSize(localMemSize);
    if (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P) {
        context->SetBlockDim(2);
        context->SetTilingKey(2);
    } else {
        /* SetBlockDim here refers to the number of cube cores, so for separated arch(AIC:AIV=1:2), 
            vector cores number is set 2 by SetDim, cube core number need to be set 1 here.*/ 
        context->SetBlockDim(1); 
        context->SetTilingKey(1);
    }

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t userWorkspaceSize = 0;
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = userWorkspaceSize + systemWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class MatmulCustom : public OpDef {
public:
    explicit MatmulCustom(const char *name) : OpDef(name)
    {
        this->Input("a")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Input("b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Output("c")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend310p")
            .AddConfig("ascend910b");
    }
};

OP_ADD(MatmulCustom);
} // namespace ops
