#!/bin/bash

set -e
# 获取脚本所在目录（RingMoE-Classification）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 1. 下载数据集
echo "Downloading dataset..."
openi dataset download tongboyuan/RS_segmentation_dataset NWPU-RESISC45.zip --cluster npu --save_path ./data

# 2. 下载模型
echo "Downloading model..."
openi model download ray4future/ringmoe ringmoe_mindir --save_path ./model

# 3. 创建模型变量目录并移动data0文件
echo "Moving model files..."
TARGET_DIR="./model/ringmoe_model_variables"
mkdir -p "$TARGET_DIR"  # 创建目标目录（如果不存在）
mv "./model/data_0" "$TARGET_DIR"  # 移动data_0文件

# 4. 解压数据集
echo "Unzipping dataset..."
unzip -q -o "./data/NWPU-RESISC45.zip" -d "./data"

# 5. 清理临时文件（可选）
echo "Cleaning up..."
rm -f "./data/NWPU-RESISC45.zip"

echo "Preparation completed successfully!"
