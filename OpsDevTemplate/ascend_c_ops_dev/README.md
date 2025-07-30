小白概念扫盲推荐阅读[昇腾Ascend TIK自定义算子开发教程（概念版）](https://blog.csdn.net/m0_37605642/article/details/132780001)

> TBE框架给用户提供自定义算子，包括TBE DSL、TBE TIK、AICPU三种开发方式，TIK用Python写算子，TIK2用c++写算子。TBE是上一代的算子开发语言了，华为TBE和AKG都基于TVM但是对动态规模的支持不是很好。目前TBE不怎么演进了，都逐步走向**AscendC**(旧名TIK C++/TIK2)。

写算子前最好理解一下[Ascend NPU的硬件架构](https://blog.csdn.net/m0_74823595/article/details/144329778)，这样能更好的理解Ascend C接口文档的涵义。

## 环境准备

**Toolkit** 开发套件 本质上包含了离线推理引擎(NNRT)和实用工具(Toolbox),所以不管是运行环境还是开发环境，只要安装了Toolkit就行。

```bash
install_path=/usr/local/Ascend/ascend-toolkit/latest 
source ${install_path}/bin/setenv.bash
export DDK_PATH=$HOME/Ascend/ascend-toolkit/latest
export NPU_HOST_LIB=$DDK_PATH/runtime/lib64/stub 
```

使用CANN运行用户编译、运行时，需要以CANN运行用户登录环境，执行 `source ${install_path}/set_env.sh`命令设置环境变量，其中 `${install_path}`为CANN软件的安装目录
`export ASCEND_INSTALL_PATH=/usr/local/Ascend可以设置环境变量以备后续使用。`

- 运行环境安装nnrt包，则开发过程中引用对应AscendCL目录。
  头文件路径：CANN软件安装后文件存储路径 `${ASCEND_INSTALL_PATH}/nnrt/latest/include/acl`
  库文件路径：CANN软件安装后文件存储路径 `${ASCEND_INSTALL_PATH}/nnrt/latest/lib64`
- 运行环境安装nnae包，则开发过程中引用对应AscendCL目录。
  头文件路径：CANN软件安装后文件存储路径 `${ASCEND_INSTALL_PATH}/nnae/latest/include/acl`
  库文件路径：CANN软件安装后文件存储路径 `${ASCEND_INSTALL_PATH}/nnae/latest/lib64`

Ascend CL（TIK2 C++）算子可用CPU模式或NPU模式执行

- CPU模式： 算子功能调试用，可以模拟在NPU上的计算行为，不需要依赖昇腾设备

```cpp
#include "tikicpulib.h"
#define_aicore_
```

- NPU模式： 算子功能/性能调试用，可以使用NPU的强大算力进行运算加速

```cpp
#include "acl/acl.h"
#define_aicore [aicore]
```

（可选）通过环境变量ASCEND_CACHE_PATH、ASCEND_WORK_PATH设置AscendCL应用编译运行过程中产生的文件的落盘路径，涉及ATC模型转换、AscendCL应用编译配置、AOE模型智能调优、性能数据采集、日志采集等功能，落盘文件包括算子编译缓存文件、知识库文件、调优结果文件、性能数据文件、日志文件等。具体工具用法和案例参考[算子场景开发工具旅程图](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha001/devaids/quickstart/devjrnmap/toolsindex_001.html)。

## Ascend C demos

一般不需要从零开始开发，checkout到对应branch的samples官方样例即可。下面以香橙派中安装CANN`8.0.RC3.alpha002 `为例

```bash
# ls /usr/local/Ascend/ascend-toolkit/
8.0  8.0.0  8.0.RC3.alpha002  latest  set_env.sh
```

动态图使用的aclnn算子清单可以查询官网，比如CANN`8.0.RC3.alpha002`的[aclnn矩阵乘法算子](https://gitee.com/ascend/samples/tree/v0.1-8.0.0.alpha002/operator/ascendc/0_introduction/10_matmul_frameworklaunch/AclNNInvocation)等。一般样例报错的话是因为Soc_version不支持，需要查询对应文档或者在仓库提需求issues，某些新版本的样例有可能适用于旧版本CANN。

为了讲解如何开发Ascend CL的算子，建议先了解一下面几个案例(来自[**https://gitee.com/ascend/samples/tree/master/operator/ascendc/tutorials**](https://gitee.com/ascend/samples/tree/master/operator/ascendc/tutorials))：
- 0.helloAscendC: [helloworld初识开发流程](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/0_helloworld);
- 1.1.vectorParadigm_kernel_invocation: 以[AddCutsom](https://gitee.com/ascend/samples/tree/v0.1-8.0.0.alpha002/operator/ascendc/0_introduction/3_add_kernellaunch/AddKernelInvocationNeo)为例，展示cpp算子<<<直调>>>的用法。核函数直调方法下，开发者完成kernel侧算子实现和host侧tiling实现后，即可通过AscendCL运行时接口，完成算子kernel直调， 该方式下tiling开发不受CANN框架的限制，简单直接，多用于算子功能的快速验证;
- 1.2.vectorParadigm_framework: 以[AddCustom](https://gitee.com/ascend/samples/tree/v0.1-8.0.0.alpha002/operator/ascendc/0_introduction/1_add_frameworklaunch/AddCustom)为例。工程调用就是打包出*.run，然后安装部署，接着被其他框架调用;
- 2.cubeParadigm： 以matmul为例，包含[基于MatmulCustom算子工程调用](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/10_matmul_frameworklaunch) 或[算子直调](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/11_matmul_kernellaunch)。
- 3.mixVectorCubeParadigm: 以融合算子[matmulleakyrelu_frameworklaunch](https://gitee.com/ascend/samples/tree/v0.1-8.0.0.alpha002/operator/ascendc/0_introduction/12_matmulleakyrelu_frameworklaunch)为例，融合算子就是矩阵计算的同时进行向量计算，总用时是$max(Time_{\ 矩阵计算模块AIC},Time_{\ 向量计算模块AIV})$。

对于frameworkLaunch开放式算子编程的方式，通过上级目录`ascend_c_ops_pybind11`和官方[PyTorch第三方框架Pybind11调用AscendC算子](https://www.hiascend.com/document/detail/zh/canncommercial/800/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0057.html)或[MindSporee框架Custom原语AOT类型自定义Ascend平台算子](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0/custom_program/operation/op_custom_ascendc.html)，实现Python调用算子kernel程序。

**PS:**：生产环境可以考虑断点保存MindSpore的算子输入数据 `mindsporeTensor.asnumpy().tofile()`，即参考[numpy输入数据导出bin文件用于AscendC算子开发的案例](https://gitee.com/ascend/samples/blob/master/cplusplus/level1_single_api/4_op_dev/6_ascendc_custom_op/kernel_invocation/MatMul/matmul_custom.py)的做法, 导出Python中的输入数据 `input_data.bin`，然后导入到Ascend C中做算子开发。类比CPU上传统的，GPU上编程模型SIMT对应SPMD执行模型可以用来实现各种高性能并行计算，AscendC语言提供“释放NPU算力去承接诸如图片渲染等大规模并行计算任务”的一种方案，比如[DVPP算子](https://bbs.huaweicloud.com/blogs/394593?utm_source=zhihu&utm_medium=bbs-ex&utm_campaign=other&utm_content=content))。如果未来Ascend NPU IR对接上Triton DSL，就可能只需要写一份python代码自定义算子，然后GPU/NPU上分别编译为CUDA/或scendC，甚至跳过这两个接口语言，直接编译到PTX之类的二进制bin直接执行。

* Ascend NPU的ISA可以用cpu孪生，CPU多线程模拟执行算子的代码跨平台执行效果最robust。CANN版本和soc_version不同很可能导致代码无法上板执行，以hiascend算子页面具体支持的产品型号为准。Ascend C自定义算子对于比较新的硬件支持情况最好，比较老的硬件则需要考虑[TIK](https://www.bilibili.com/video/BV1ha4y1V7vK)、[AICPU](https://www.bilibili.com/video/BV1qg41167db/)和[TBE DSL自定义算子](https://www.bilibili.com/video/BV17v4y1D7K5/)。

* 更多理论介绍参考《Ascend C异构并行程序设计：昇腾算子编程指南》，更多代码实践参考Ascend仓库samples代码。

