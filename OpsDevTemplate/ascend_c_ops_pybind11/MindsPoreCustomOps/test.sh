#!/bin/bash
script_dir=$(dirname "$0")
export ASCEND_CUSTOM_OPP_PATH="/root/HeterogeneousComputingOpsDevTemplate/ascend_c_ops_dev/1.2.vectorParadigm_framework/AddCustom/myinstallpath/packages/vendors/customize":${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=/root/HeterogeneousComputingOpsDevTemplate/ascend_c_ops_dev/1.2.vectorParadigm_framework/AddCustom/myinstallpath/packages/vendors/customize/op_api/lib:${LD_LIBRARY_PATH}
python ./tests/test_add_custom_aclnn_can_be_disapear.py  
python ./tests/test_customize_aclnnAddCustom.py


