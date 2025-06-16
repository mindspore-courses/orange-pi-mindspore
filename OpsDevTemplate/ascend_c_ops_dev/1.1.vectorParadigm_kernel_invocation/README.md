https://gitee.com/ascend/samples/tree/v0.2-8.0.0.beta1/cplusplus/level1_single_api/4_op_dev/6_ascendc_custom_op/kernel_invocation/Add

```bash
cd ./Add
bash run.sh add_custom ascend910 AiCore cpu # 没有Ascend NPU也能模拟执行
bash run.sh add_custom ascend910b AiCore npu # 需要有对应设备 [Ascend310P1 Ascend310P3 Ascend910B1 Ascend910B2 Ascend910B3 Ascend910B4]方可支持
```

会拉起模拟线程然后sim执行，最后得到仿真结果：

```bash
[100%] Built target add_custom_cpu
/home/shared/OperatorsDevTemplate/ascend_c_ops/ascend_addd_kernel_invocation
INFO: compile op on cpu succeed!
[DEBUG] config_file.cc:157 getConfigPath Config file is found, path is /usr/local/Ascend/ascend-toolkit/8.0.RC3.alpha001/aarch64-linux/simulator/Ascend910A/lib/config_pv_aicore_model.toml
[DEBUG] config_file.cc:157 getConfigPath Config file is found, path is /usr/local/Ascend/ascend-toolkit/8.0.RC3.alpha001/aarch64-linux/simulator/Ascend910A/lib/config_pv_aicore_model.toml
[DEBUG] config_file.cc:157 getConfigPath Config file is found, path is /usr/local/Ascend/ascend-toolkit/8.0.RC3.alpha001/aarch64-linux/simulator/Ascend910A/lib/config_pv_aicore_model.toml
[DEBUG] config_file.cc:157 getConfigPath Config file is found, path is /usr/local/Ascend/ascend-toolkit/8.0.RC3.alpha001/aarch64-linux/simulator/Ascend910A/lib/config_pv_aicore_model.toml
[DEBUG] config_file.cc:157 getConfigPath Config file is found, path is /usr/local/Ascend/ascend-toolkit/8.0.RC3.alpha001/aarch64-linux/simulator/Ascend910A/lib/config_pv_aicore_model.toml
[DEBUG] config_file.cc:157 getConfigPath Config file is found, path is /usr/local/Ascend/ascend-toolkit/8.0.RC3.alpha001/aarch64-linux/simulator/Ascend910A/lib/config_pv_aicore_model.toml
[DEBUG] config_file.cc:157 getConfigPath Config file is found, path is /usr/local/Ascend/ascend-toolkit/8.0.RC3.alpha001/aarch64-linux/simulator/Ascend910A/lib/config_pv_aicore_model.toml
[DEBUG] config_file.cc:157 getConfigPath Config file is found, path is /usr/local/Ascend/ascend-toolkit/8.0.RC3.alpha001/aarch64-linux/simulator/Ascend910A/lib/config_pv_aicore_model.toml
[SUCCESS][CORE_0][pid 806095] exit success!
[SUCCESS][CORE_1][pid 806096] exit success!
[SUCCESS][CORE_2][pid 806097] exit success!
[SUCCESS][CORE_3][pid 806098] exit success!
[SUCCESS][CORE_4][pid 806099] exit success!
[SUCCESS][CORE_5][pid 806100] exit success!
[SUCCESS][CORE_6][pid 806101] exit success!
[SUCCESS][CORE_7][pid 806102] exit success!
INFO: execute op on cpu succeed!
md5sum: 
94ca48a53a841e4d7bb603ef88dc1dcd  output/golden.bin
94ca48a53a841e4d7bb603ef88dc1dcd  output/output_z.bin
```

哈希相同代表生成的结果是正确的，和numpy计算的向量和一摸一样。
