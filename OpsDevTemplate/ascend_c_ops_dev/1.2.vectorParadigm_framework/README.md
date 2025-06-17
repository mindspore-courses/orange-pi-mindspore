```bash
bash ./build.sh
./build_out/custom_opp_ubuntu_aarch64.run --extract="$pwd/myinstallpath"
```

```bash
export ASCEND_CUSTOM_OPP_PATH=/root/HeterogeneousComputingOpsDevTemplate/ascend_c_ops_dev/1.2.vectorParadigm_framework/AddCustom/myinstallpath/packages/vendors/customize:${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=/root/HeterogeneousComputingOpsDevTemplate/ascend_c_ops_dev/1.2.vectorParadigm_framework/AddCustom/myinstallpath/packages/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}
###/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/  
```

```bash
ascend_c_ops/MindsPoreCustomOps/ascendcSample4AddCustom/myinstallpath/vendors/customize# nm ./op_api/lib/libcust_opapi.so -D | grep aclnn
0000000000001434 T aclnnAddCustom
00000000000010f0 T aclnnAddCustomGetWorkspaceSize
```

