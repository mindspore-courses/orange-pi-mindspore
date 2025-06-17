示例如下。
```bash
bash run.sh -v Ascend310B4
```

---

## HelloWorld自定义算子样例说明
<!--注：该样例仅用于说明目的，不用作生产质量代码的示例-->
本样例通过使用<<<>>>内核调用符完成算子核函数NPU侧运行验证的基础流程和PRINTF宏打印。

## 支持的产品型号
样例支持的产品型号为：
- Atlas 推理系列产品（Ascend 310P处理器）
- Atlas A2训练系列产品/Atlas 800I A2推理产品
- Atlas 200/500 A2推理产品

## 目录结构介绍
```
├── HelloWorldSample
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── hello_world.cpp         // 算子kernel实现
│   ├── main.cpp                // 主函数，调用算子的应用程序，含CPU域及NPU域调用
│   └── run.sh                  // 编译运行算子的脚本
```

## 环境要求
编译运行此样例前，请参考[《CANN软件安装指南》](https://hiascend.com/document/redirect/CannCommunityInstSoftware)完成开发运行环境的部署。

## 编译运行样例算子

### 1.准备：获取样例代码

 可以使用以下两种方式下载，请选择其中一种进行源码准备。

 - 命令行方式下载（下载时间较长，但步骤简单）。

   ```bash
   # 开发环境，非root用户命令行中执行以下命令下载源码仓。git_clone_path为用户自己创建的某个目录。
   cd ${git_clone_path}
   git clone https://gitee.com/ascend/samples.git
   ```
   **注：如果需要切换到其它tag版本，以v0.5.0为例，可执行以下命令。**
   ```bash
   git checkout v0.5.0
   ```
 - 压缩包方式下载（下载时间较短，但步骤稍微复杂）。

   **注：如果需要下载其它版本代码，请先请根据前置条件说明进行samples仓分支切换。**
   ```bash
   # 1. samples仓右上角选择 【克隆/下载】 下拉框并选择 【下载ZIP】。
   # 2. 将ZIP包上传到开发环境中的普通用户某个目录中，【例如：${git_clone_path}/ascend-samples-master.zip】。
   # 3. 开发环境中，执行以下命令，解压zip包。
   cd ${git_clone_path}
   unzip ascend-samples-master.zip
   ```

### 2.编译运行样例工程

  - 打开样例目录

    ```bash
    cd ${git_clone_path}/samples/operator/HelloWorldSample
    ```

  - 配置修改
    * 修改CMakeLists.txt内SOC_VERSION为所需产品型号
    * 修改CMakeLists.txt内ASCEND_CANN_PACKAGE_PATH为CANN包的安装路径

  - 样例执行

    ```bash
    bash run.sh -v [SOC_VERSION]
    ```
    - SOC_VERSION：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行
      npu-smi info命令进行查询，在查询到的“Name”前增加Ascend信息，例如“Name”对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。支持以下参数取值（xxx请替换为具体取值）：
      - Atlas 推理系列产品（Ascend 310P处理器）参数值：Ascend310P1、Ascend310P3
      - Atlas A2训练系列产品/Atlas 800I A2推理产品参数值：AscendxxxB1、AscendxxxB2、AscendxxxB3、AscendxxxB4
      - Atlas 200/500 A2推理产品参数值：AscendxxxB1、AscendxxxB2、AscendxxxB3、AscendxxxB4

## 更新说明
| 时间       | 更新事项                                     |
| ---------- | -------------------------------------------- |
| 2023/10/23 | 新增HelloWorldSample样例                     |
| 2024/03/07 | 修改样例编译方式，并添加PRINTF宏使用方式展示 |
| 2024/05/16 | 修改readme结构，新增目录结构                 |

## 已知issue

  暂无
