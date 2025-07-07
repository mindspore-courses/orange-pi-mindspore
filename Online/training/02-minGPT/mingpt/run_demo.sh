# 因为香橙派不支持all_finite算子，切换为小算子实现
export MS_DEV_RUNTIME_CONF=all_finite:false
python demo.py