# mindspore.mint接口测试

## 测试说明
测试脚本：test_mint.py
**注意：** 该脚本测试mint接口acl算子执行情况，仅测试算子是否支持，不涉及精度测试

### 测试依赖
- numpy
- pytest
- mindspore 2.5.0(需要测试的版本)

### 测试命令
1. 整体测试
```shell
    pytest -vs test_mint.py
```
2. 单个类测试（如测试TensorCreationOperationsTest类）
```shell
    pytest -vs test_mint.py::TensorCreationOperationsTest
```
3. 单个方法测试（如测试TensorCreationOperationsTest类中的test_arange方法）
```shell
    pytest -vs test_mint.py::TensorCreationOperationsTest::test_arange
```
4. 日志重定向（将日志输出到文件中）
```shell
    pytest -vs test_mint.py > test.log 2>&1
```
