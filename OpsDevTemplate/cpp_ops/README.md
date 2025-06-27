```bash

# 1.先切换回项目的根目录
# 2.
mkdir build &&  cd build
# 3.-DPYTHON_EXECUTABLE是python的路径。编译的时候用的python环境在后续调用需要相同的conda环境
# cmake .. -DPYTHON_EXECUTABLE=/Users/username/miniconda3/bin/python
cmake .. -DPYTHON_EXECUTABLE=$(which python)
# 4.
cd ..
make all
file my_mindspore_ops.cpython-312-x86_64-linux-gnu.so
python setup.py build_ext --inplace
# 然后就生成了一个.so文件
python setup.py bdist_wheel
# 创建whl文件

```

GEMM通用矩阵相乘是比较常见的的算子，这里有讲解[如何使用x86的SSE/AVX指令给gemm加速](https://lzzmm.github.io/2021/09/10/GEMM/), 其他平台的SIMD指令集还有： arm64的NEON指令和risc-v的RV指令。
