import os
import numpy as np
from ais_bench.infer.interface import InferSession
from collections import defaultdict
import time
import glob
import mindspore as ms
import mindspore.dataset as de
import mindspore.common.dtype as mstype
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.dataset.vision.c_transforms as C
from mindspore.dataset.vision import Inter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 自动构建路径 - 使用相对路径
MODEL_PATH = os.path.join(SCRIPT_DIR, "model", "ringmoe_classification.om")
DATA_ROOT = os.path.join(SCRIPT_DIR, "data", "NWPU-RESISC45", "test")

BATCH_SIZE = 4
CLASS_MAP = {
    0: "airplane",
    1: "airport",
    2: "baseball_diamond",
    3: "basketball_court",
    4: "beach",
    5: "bridge",
    6: "chapel",
    7: "church",
    8: "circular_farmland",
    9: "cloud",
    10: "commercial_area",
    11: "dense_residential",
    12: "desert",
    13: "forest",
    14: "freeway",
    15: "golf_course",
    16: "ground_track_field",
    17: "harbor",
    18: "industrial_area",
    19: "intersection",
    20: "island",
    21: "lake",
    22: "meadow",
    23: "medium_residential",
    24: "mobile_home_park",
    25: "mountain",
    26: "overpass",
    27: "palace",
    28: "parking_lot",
    29: "railway",
    30: "railway_station",
    31: "rectangular_farmland",
    32: "river",
    33: "roundabout",
    34: "runway",
    35: "sea_ice",
    36: "ship",
    37: "scrubland",
    38: "sparse_residential",
    39: "stadium",
    40: "storage_tank",
    41: "tennis_court",
    42: "terrace",
    43: "thermal_power_station",
    44: "wetland"
}

# 检查路径是否存在
def check_path(path, name):
    if not os.path.exists(path):
        print(f"错误: {name}路径不存在 - {path}")
        print("请检查路径设置并确保文件/目录存在")
        exit(1)
    else:
        print(f"{name}路径验证成功: {path}")

# 初始化模型前先检查路径
check_path(MODEL_PATH, "模型")
check_path(DATA_ROOT, "数据集")

# 初始化模型
print("正在加载模型...")
start_time = time.time()
session = InferSession(device_id=0, model_path=MODEL_PATH)
print(f"模型加载完成，耗时: {time.time()-start_time:.2f}秒")

# 获取模型输入信息
input_info = session.get_inputs()[0]
print("\n模型输入信息:")
print(f"  形状: {input_info.shape}")

# 配置类
class Config:
    def __init__(self):
        self.finetune_dataset = self.DatasetConfig()
        self.seed = 0
        self.auto_tune = True
        self.profile = False
        self.filepath_prefix = "./autotune"
        self.autotune_per_step = 10
    
    class DatasetConfig:
        def __init__(self):
            self.eval_path = DATA_ROOT
            self.image_size = 224
            self.interpolation = "BICUBIC"
            self.input_columns = ["image", "label"]
            self.num_workers = 2
            self.python_multiprocessing = True
            self.prefetch_size = 30
            self.numa_enable = False
            self.batch_size = BATCH_SIZE
            self.repeat = 1
            self.device_num = 1
            self.local_rank = 0
            self.samples_num = 0

# 从云端代码复制的函数
def build_dataset(config, is_train=True):
    if is_train:
        return None
    else:
        # 再次检查数据路径
        if not os.path.exists(config.eval_path):
            print(f"错误: 数据集路径不存在 - {config.eval_path}")
            exit(1)
            
        image_count = 0
        for class_dir in os.listdir(config.eval_path):
            class_path = os.path.join(config.eval_path, class_dir)
            if os.path.isdir(class_path):
                image_files = glob.glob(os.path.join(class_path, "*.jpg")) + \
                             glob.glob(os.path.join(class_path, "*.jpeg")) + \
                             glob.glob(os.path.join(class_path, "*.png"))
                image_count += len(image_files)
        
        config.samples_num = image_count
        
        print(f"找到 {image_count} 张图像")
        
        ds = de.ImageFolderDataset(config.eval_path,
                                   num_parallel_workers=config.num_workers,
                                   shuffle=False,
                                   num_shards=config.device_num,
                                   shard_id=config.local_rank)
        print(f"评估数据集大小: {ds.get_dataset_size()}")
        return ds

def build_transforms(config, interpolation, is_train=True):
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    trans = [
        C.Decode(),
        C.Resize(int(256 / 224 * config.image_size), interpolation=interpolation),
        C.CenterCrop(config.image_size),
        C.Normalize(mean=mean, std=std),
        C.HWC2CHW()
    ]
    return trans

def create_finetune_dataset(config, is_train=True):
    if hasattr(Inter, config.finetune_dataset.interpolation):
        interpolation = getattr(Inter, config.finetune_dataset.interpolation)
    else:
        interpolation = Inter.BICUBIC

    ds = build_dataset(config.finetune_dataset, is_train)
    transforms = build_transforms(config.finetune_dataset, interpolation, is_train)
    
    type_cast_op = C2.TypeCast(mstype.int8)
    ds = ds.map(input_columns=config.finetune_dataset.input_columns[0],
                num_parallel_workers=config.finetune_dataset.num_workers,
                operations=transforms,
                python_multiprocessing=config.finetune_dataset.python_multiprocessing)
    ds = ds.map(input_columns=config.finetune_dataset.input_columns[1],
                num_parallel_workers=config.finetune_dataset.num_workers,
                operations=type_cast_op)
    
    ds = ds.batch(config.finetune_dataset.batch_size, drop_remainder=False)
    ds = ds.repeat(config.finetune_dataset.repeat)
    return ds

# 创建配置实例
config = Config()

# 创建数据集
print("\n创建评估数据集...")
dataset = create_finetune_dataset(config, is_train=False)

# 创建迭代器
data_iterator = dataset.create_dict_iterator(output_numpy=True)

# 初始化统计变量
confusion_matrix = defaultdict(lambda: defaultdict(int))
class_stats = defaultdict(lambda: {"correct": 0, "total": 0})
total_samples = 0
correct_predictions = 0
failed_images = []

# 定义NumPy操作
def argmax(x):
    return np.argmax(x, axis=-1)

def cast(x, dtype):
    if dtype == np.int32:
        return x.astype(np.int32)
    elif dtype == np.float32:
        return x.astype(np.float32)
    return x

def equal(a, b):
    return np.equal(a, b)

def reduce_sum(x):
    return np.sum(x)

# 批量处理所有图像
print("\n开始批量推理处理...")
start_time = time.time()
batch_count = dataset.get_dataset_size()

# 最大连续失败次数
MAX_CONSECUTIVE_FAILURES = 5
consecutive_failures = 0

for batch_idx, batch in enumerate(data_iterator):
    images = batch["image"]
    labels = batch["label"]
    actual_batch_size = images.shape[0]
    
    if (batch_idx + 1) % 10 == 0 or batch_idx == batch_count - 1:
        print(f"处理批次 [{batch_idx+1}/{batch_count}] - 图像 {batch_idx*BATCH_SIZE+1} 到 {batch_idx*BATCH_SIZE+actual_batch_size}")
    
    try:
        # 执行推理
        outputs = session.infer([images])
        
        if not outputs:
            print(f"无输出结果: 批次 {batch_idx+1}")
            failed_images.extend([f"batch_{batch_idx}_idx_{i}" for i in range(actual_batch_size)])
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"连续失败次数超过阈值({MAX_CONSECUTIVE_FAILURES})，停止推理")
                break
            continue
        
        logits = outputs[0]  # 分类logits
        
        # 使用NumPy模拟计算逻辑
        y_pred = argmax(logits)           # 获取预测类别
        y_pred = cast(y_pred, np.int32)   # 转换为int32
        y_true = cast(labels, np.int32)   # 确保标签为int32
        
        # 计算正确预测的数量
        y_correct = equal(y_pred, y_true)
        y_correct = cast(y_correct, np.float32)
        batch_correct = reduce_sum(y_correct)
        
        # 更新总正确数
        correct_predictions += batch_correct
        
        total_samples += actual_batch_size
        consecutive_failures = 0  # 重置连续失败计数
        
    except Exception as e:
        print(f"批量推理失败: 批次 {batch_idx+1} - {str(e)}")
        import traceback
        traceback.print_exc()
        failed_images.extend([f"batch_{batch_idx}_idx_{i}" for i in range(actual_batch_size)])
        consecutive_failures += 1
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            print(f"连续失败次数超过阈值({MAX_CONSECUTIVE_FAILURES})，停止推理")
            break

# 计算总耗时
total_time = time.time() - start_time

# 计算整体准确率
overall_acc = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0

# 修复：确保使用整数类别ID
for cls_id in sorted(class_stats.keys()):
    stats = class_stats[cls_id]
    if stats["total"] > 0:
        acc = stats["correct"] / stats["total"] * 100
        class_name = CLASS_MAP.get(cls_id, f"未知类别_{cls_id}")
        print(f"{class_name:<25} {acc:.2f}%     {stats['correct']}/{stats['total']}")

print("\n" + "=" * 50)
print(f"整体准确率: {overall_acc:.2f}% ({correct_predictions}/{total_samples})")
print(f"总图像数: {config.finetune_dataset.samples_num}, 成功处理: {total_samples}, 失败: {len(failed_images)}")

if failed_images:
    print(f"\n警告: {len(failed_images)} 张图像处理失败")
    with open("failed_images.txt", "w") as f:
        f.write("\n".join(failed_images))

# 性能分析
if total_samples > 0:
    avg_time = total_time / total_samples
    print(f"\n总处理时间: {total_time:.2f}秒")
    print(f"平均每张图像处理时间: {avg_time*1000:.2f}ms")
    print(f"整体吞吐量: {total_samples/total_time:.2f} FPS")
else:
    print("\n警告: 没有成功处理任何样本，无法计算性能指标")
