/**
 * @file add_custom.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "add_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    //**数据切分函数,正规的tiling需要通过context，算子直调测试的时候才不需要 */
    TilingData tiling;
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize(); // 输入的总长度
    context->SetBlockDim(BLOCK_DIM); // 设置每个核的blockDim为8
    tiling.set_totalLength(totalLength); // 设置总长度
    tiling.set_tileNum(TILE_NUM);//设置每个核计算多少块
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
    /**1. 形状推导
    2. 数据类型推导
    3. 数据排布格式推导 **/
static graphStatus InferShape(gert::InferShapeContext *context)
{
    /**Shape推导 */
    const gert::Shape *x1_shape = context->GetInputShape(0); // 获取输入的形状信息
    gert::Shape *y_shape = context->GetOutputShape(0); // 获取输出的形状信息
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class AddCustom : public OpDef {
public:
    explicit AddCustom(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType); // 设置前面个定义的Shape推导函数
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910")
            .AddConfig("ascend310p")
            .AddConfig("ascend310b")
            .AddConfig("ascend910b");// 绑定不同版本的Ascend NPU!
    }
};
OP_ADD(AddCustom);
} // namespace ops
