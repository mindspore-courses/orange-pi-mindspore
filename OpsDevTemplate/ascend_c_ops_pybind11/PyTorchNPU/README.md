## 目录结构介绍

```
├── CppExtensions
│   ├── add_custom_test.py      // python调用脚本
│   ├── add_custom.cpp          // 算子实现
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── pybind11.cpp            // pybind11函数封装
│   └── run.sh                  // 编译运行算子的脚本
```

## 代码实现介绍

- kernel实现Add算子的数学表达式为：

  ```
  z = x + y
  ```

  计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储，然后使用计算接口完成两个输入参数相加，得到最终结果，再搬出到外部存储上。

  Add算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入Tensor xGm和yGm搬运到Local Memory，分别存储在xLocal、yLocal，Compute任务负责对xLocal、yLocal执行加法操作，计算结果存储在zLocal中，CopyOut任务负责将输出数据从zLocal搬运至Global Memory上的输出Tensor zGm中。具体请参考[add_custom.cpp](./add_custom.cpp)。
- 调用实现
  通过PyTorch框架进行模型的训练、推理时，会调用到很多算子进行计算，调用方式也和kernel编译流程相关。对于自定义算子工程，需要使用PyTorch Ascend Adapter中的OP-Plugin算子插件对功能进行扩展，让torch可以直接调用自定义算子包中的算子；对于KernelLaunch开放式算子编程的方式，也可以使用pytorch调用，此样例演示的就是这种算子调用方式。

  pybind11.cpp文件是一个C++的代码示例，使用了pybind11库来将C++代码封装成Python模块。该代码实现中定义了一个名为m的pybind11模块，其中包含一个名为run_add_custom的函数。该函数与my_add::run_add_custom函数相同，用于将C++函数转成Python函数。在函数实现中，通过c10_npu::getCurrentNPUStream() 的函数获取当前NPU上的流，并调用ACLRT_LAUNCH_KERNEL宏启动自定义的Kernel函数add_custom，在NPU上执行算子。

  在add_custom_test.py调用脚本中，通过导入自定义模块add_custom，调用自定义模块add_custom中的run_add_custom函数，在NPU上执行x和y的加法操作，并将结果保存在变量z中。

## 运行样例算子

- 安装pytorch (这里使用2.1.0版本为例)

  **aarch64:**

  ```bash
  pip3 install torch==2.1.0
  ```

  **x86:**

  ```bash
  pip3 install torch==2.1.0+cpu  --index-url https://download.pytorch.org/whl/cpu
  ```
- 安装torch-npu （以Pytorch2.1.0、python3.9、CANN版本8.0.RC1.alpha002为例）

  ```bash
  git clone https://gitee.com/ascend/pytorch.git -b v6.0.rc1.alpha002-pytorch2.1.0
  cd pytorch/
  bash ci/build.sh --python=3.9
  pip3 install dist/*.whl
  ```

  安装pybind11

  ```bash
  pip3 install pybind11
  ```
- 打开样例目录以命令行方式下载样例代码，master分支为例。

  ```bash
  cd ${git_clone_path}/samples/operator/AddCustomSample/KernelLaunch/CppExtensions
  ```
- 修改配置

  * 修改CMakeLists.txt内SOC_VERSION为所需产品型号。
  * 修改CMakeLists.txt内ASCEND_CANN_PACKAGE_PATH为CANN包的安装路径。
  * 修改CMakeLists.txt内RUN_MODE为所需编译模式。

  RUN_MODE：编译方式，当前仅支持NPU上板。支持参数为[npu]，默认值为npu。

  SOC_VERSION：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的“Name”前增加Ascend信息，例如“Name”对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。支持以下参数取值（xxx请替换为具体取值）：

  - Atlas 训练系列产品参数值：AscendxxxA、AscendxxxB
  - Atlas 推理系列产品（Ascend 310P处理器）参数值：Ascend310P1、Ascend310P3
  - Atlas A2训练系列产品/Atlas 800I A2推理产品参数值：AscendxxxB1、AscendxxxB2、AscendxxxB3、AscendxxxB4
  - Atlas 200/500 A2推理产品参数值：AscendxxxB1、AscendxxxB2、AscendxxxB3、AscendxxxB4
- 样例执行（推荐用法）

  ```bash
  rm -rf build
  mkdir build
  cd build
  cmake ..
  make
  python3 ../add_custom_test.py
  ```

  用户亦可参考run.sh脚本进行编译与运行。

  ```bash
  bash run.sh -v ascend310B4 # conda环境会有冲突，编译时使用默认的解析器和运行时的会不一致
  ```

ascend910b1;ascend910b2;ascend910b2c;ascend910b3;ascend910b4;ascend910b4-;
ascend910_9391;ascend910_9381;ascend910_9372;ascend910_9392;ascend910_9382;ascend910_9361;
ascend910a;ascend910proa;
ascend910b;ascend910prob;ascend910premiuma;
ascend310p1;ascend310p3;
ascend310p3vir01;ascend310p3vir02;ascend310p3vir04;ascend310p3vir08
## 更新说明

| 时间       | 更新事项     |
| ---------- | ------------ |
| 2024/05/22 | 新增本readme |
