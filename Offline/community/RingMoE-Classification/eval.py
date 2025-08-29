import numpy as np
import mindspore_lite as mslite
from mindspore import context
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import os
import time
import sys
from tqdm import tqdm

# 设置环境变量 - 优化NPU运行
os.environ['TUNE_OPS_ONLINE'] = '0'
os.environ['MS_BUILD_PROCESS_NUM'] = '1'
os.environ['OPTION_EXEC_PARTITION_OPS'] = '1'
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '3'
os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'  # 减少日志输出

def create_dataset(dataset_path, batch_size=4):
    """创建数据集（优化版）"""
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    
    transform = [
        vision.Decode(),
        vision.Resize(256),
        vision.CenterCrop(224),
        vision.Normalize(mean=mean, std=std),
        vision.HWC2CHW()
    ]

    dataset = ds.ImageFolderDataset(
        dataset_path,
        num_parallel_workers=4,
        shuffle=False
    )

    dataset = dataset.map(
        operations=transform,
        input_columns="image"
    )

    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 构建路径
    mindir_path = os.path.join(SCRIPT_DIR, "model", "ringmoe_model_graph.mindir")
    dataset_path = os.path.join(SCRIPT_DIR, "data", "NWPU-RESISC45", "test")
    batch_size = 4

    # 1. 初始化Lite上下文 - 优化配置
    lite_context = mslite.Context()
    lite_context.target = ["ascend"]
    lite_context.ascend.device_id = 0
    lite_context.ascend.precision_mode = "preferred_optimal"
    lite_context.ascend.auto_tune_mode = "RL,GA"
    lite_context.ascend.provider = "ge"
    
    # 2. 加载模型 - 添加重试机制
    max_retries = 3
    model = None
    
    for attempt in range(max_retries):
        try:
            model = mslite.Model()
            print(f"\n🔄 正在编译模型(尝试 {attempt+1}/{max_retries})...")
            start_compile = time.time()
            model.build_from_file(mindir_path, mslite.ModelType.MINDIR, lite_context)
            compile_time = time.time() - start_compile
            print(f"✅ 模型成功加载! 编译时间: {compile_time:.2f}秒")
            
            # 打印输入输出信息
            inputs = model.get_inputs()
            outputs = model.get_outputs()
            print(f"输入数量: {len(inputs)}")
            for i, inp in enumerate(inputs):
                print(f"  输入[{i}]: 形状={list(inp.shape)}, 数据类型={inp.dtype}")
            print(f"输出数量: {len(outputs)}")
            for i, out in enumerate(outputs):
                print(f"  输出[{i}]: 形状={list(out.shape)}, 数据类型={out.dtype}")
            break
                
        except Exception as e:
            print(f"❌ 尝试 {attempt+1} 失败: {str(e)}")
            if attempt == max_retries - 1:
                print("⛔ 模型加载失败，请检查:")
                print(f"1. 模型路径: {mindir_path}")
                print(f"2. Ascend环境: npu-smi 是否正常工作")
                print(f"3. 模型是否兼容当前Ascend版本")
                return
            time.sleep(5)  # 等待后重试

    # 3. 创建数据集
    try:
        dataset = create_dataset(dataset_path, batch_size)
        n_batches = dataset.get_dataset_size()
        total_samples = n_batches * batch_size
        print(f"\n📊 数据集信息:")
        print(f" 批次大小: {batch_size}")
        print(f" 总批次: {n_batches}")
        print(f" 样本数量: {total_samples}")
    except Exception as e:
        print(f"❌ 数据集加载失败: {str(e)}")
        return

    # 4. 推理执行 - 优化形状处理
    correct_predictions = 0
    total_time = 0
    processed_samples = 0

    print("\n🚀 开始推理...")
    progress_bar = tqdm(total=n_batches, desc="推理进度", file=sys.stdout)

    for batch_idx, (images, labels) in enumerate(dataset.create_tuple_iterator()):
        try:
            # 转换并确保输入数据格式正确
            images_np = np.array(images.asnumpy(), dtype=np.float16)  # 显式转换为numpy数组
            
            # 获取输入Tensor并准备数据
            input_tensor = model.get_inputs()[0]
            
            # 更健壮的形状处理
            if images_np.shape != tuple(input_tensor.shape):
                images_np = np.reshape(images_np, input_tensor.shape)
            
            input_tensor.set_data_from_numpy(images_np)
            
            # 推理执行
            start_time = time.time()
            outputs = model.predict([input_tensor])
            batch_time = (time.time() - start_time) * 1000
            total_time += batch_time
            
            # 处理输出
            output_data = outputs[0].get_data_to_numpy()
            preds = np.argmax(output_data, axis=1)
            labels_np = labels.asnumpy()
            
            # 更新指标
            batch_correct = np.sum(preds == labels_np)
            correct_predictions += batch_correct
            processed_samples += len(labels_np)
            
            progress_bar.update(1)
            progress_bar.set_postfix({
                "准确率": f"{correct_predictions/processed_samples:.4f}",
                "速度": f"{batch_time:.2f}ms/批次"
            })
            
        except Exception as e:
            print(f"\n⚠️ 批次 {batch_idx+1} 错误: {str(e)}")
            continue

    progress_bar.close()

    # 5. 结果统计
    if processed_samples > 0:
        final_accuracy = correct_predictions / processed_samples
        avg_time = total_time / processed_samples
        throughput = processed_samples / (total_time / 1000)
    else:
        final_accuracy = avg_time = throughput = 0

    print("\n📊 最终结果:")
    print(f"处理样本: {processed_samples}/{total_samples}")
    print(f"正确预测: {correct_predictions}")
    print(f"准确率: {final_accuracy:.4f}")
    print(f"平均耗时: {avg_time:.2f}ms/样本")
    print(f"吞吐量: {throughput:.2f} 样本/秒")

if __name__ == "__main__":
    main()
