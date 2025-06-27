cat /usr/local/Ascend/version.info
cat /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info
cat /usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/kernel/version.info
python -c "import acl;"
cat /usr/local/Ascend/firmware/version.info
cat /usr/local/Ascend/driver/version.info
#/home/ms/mindspore/build/package$ pip install mindspore-2.6.0rc1-cp310-cp310-linux_aarch64.whl --force
pip list | grep mindspore
cat $(pip show mindspore | grep Location | awk '{print $2}')/mindspore/.commit_id
python -c "import mindspore;mindspore.run_check()"
# 检测当前Ascend环境的脚本
npu-smi info
