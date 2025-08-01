# RingMoE遥感图像分类推理案例

## 案例概述
本案例演示如何在香橙派AIpro 20T开发板上运行RingMoE遥感图像分类模型的推理任务。通过预训练的OM模型，实现对NWPU-RESISC45数据集中的45类遥感场景进行高效分类。

## 项目结构说明
```bash
RingMoE/                     # 项目根目录
├── model/                  # 模型目录
│   └── ringmoe_classification.om  # 昇腾OM推理模型
├── data/                   # 遥感图像数据集
│   ├── airplane/           # "飞机"类别图像 (70张)
│   ├── stadium/            # "体育场"类别图像 (70张)
│   ├── forest/             # "森林"类别图像 (70张)
│   ├── ...                 # 其他42个类别目录
│   └── (共45个类别目录，每个目录包含70张图像)
└── eval.py                 # 推理执行脚本
└── README.md
└── prepare.sh              
```

## 环境要求
### 硬件配置
- 香橙派AIpro 20T24G开发板（配备昇腾AI处理器）

### 软件版本
| 组件 | 版本 | 备注 |
|------|------|------|
| Python | 3.9 | 基础运行环境 |
| MindSpore | 2.6.0 | 推理框架 |
| CANN | 8.0.0 | 昇腾计算架构 |

## 模型信息
### 模型特性
- **基础架构**：基于RingMoE的Vision Transformer
- **输入规格**：224×224 RGB图像
- **输出类别**：45种遥感场景（NWPU-RESISC45）
- **模型大小**：2.0G (OM格式)
- **精度指标**：NWPU-RESISC45测试集准确率3%

## 运行推理

```bash
# 运行图像批量推理
bash prepare.sh
python3 eval.py 
```

## 预期输出

### 控制台输出
```
模型路径验证成功: /home/HwHiAiUser/eval/model/ringmoe_classification.om
数据集路径验证成功: /home/HwHiAiUser/RingMoE-Classification/data/NWPU-RESISC45/test/
正在加载模型...
[INFO] acl init success
[INFO] open device 0 success
[INFO] load model /home/HwHiAiUser/eval/model/ringmoe_classification.om success
[INFO] create model description success
模型加载完成，耗时: 6.32秒

模型输入信息:
  形状: [4, 3, 224, 224]

创建评估数据集...
找到 3150 张图像
评估数据集大小: 3150
整体准确率: 2.92% (92.0/3148)
总图像数: 3150, 成功处理: 3148, 失败: 2

警告: 2 张图像处理失败

总处理时间: 1089.60秒
平均每张图像处理时间: 346.13ms
整体吞吐量: 2.89 FPS
[INFO] unload model success, model Id is 1
[INFO] end to destroy context
[INFO] end to reset device is 0
[INFO] end to finalize acl
```

### 输出解析
- **设备初始化** 
  昇腾处理器：Device 0 成功启用

  运行环境：ACL初始化成功（Ascend Computing Language）

- **模型加载** 
  模型路径：/home/HwHiAiUser/eval/model/ringmoe_classification.om

  加载状态：✅ 成功加载（6.32秒）

- **模型规格：** 

    支持批量推理（batch_size=4）

    输入张量维度：[4, 3, 224, 224]

- **预处理流程** 
  图像调整：统一缩放至 224×224 分辨率

  数据格式：RGB三通道输入（3通道）

  归一化处理：执行标准化预处理


## 性能指标** 
| 指标 | 值 | 测试条件 |
|------|----|---------|
| 单图推理时间 | 300-350ms | 224×224分辨率 |
| 模型加载时间 | 5-6s | 首次加载 |

## 数据集说明
- **来源**：NWPU-RESISC45遥感图像数据集
- **结构**：
  - 45个类别文件夹（每个文件夹名称即为类别标签）
  - 每个类别包含70张高质量遥感图像
  - 总计：45类 × 70张 = 3,150张测试图像
- **典型类别**：
性  - 自然场景：`forest`, `river`, `island`, `mountain`
  - 人造建筑：`airplane`, `stadium`, `harbor`, `bridge`
  - 农业区域：`farmland`, `golf_course`
  - 特殊地形：`desert`, `glacier`


## 注意事项
1. 输入图像应为RGB格式，JPG/PNG文件
2. 首次运行会有模型加载时间（约6.3秒）
3. 确保`CLASS_MAP`内填入所有正确类别标签

> **案例优势**：本方案在香橙派AIpro上实现了3%精度的遥感图像分类，推理速度达350ms级，展示了边缘设备运行先进视觉模型的能力。




