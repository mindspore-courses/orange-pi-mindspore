import numpy as np

from dofbot_garbage_yolov5.utils.garbage_identify import GarbageIdentify


class TestGarbageIdentify:
    def setup_method(self):
        self.img = np.zeros((640, 640, 3), dtype=np.uint8)
        self.garbage_identify = GarbageIdentify(test_mode=True)

    #### 跑以下测试用例时，需要先开启ros server
    # 测试较小index时，garbage_run函数输出类型
    def test_garbage_run(self):
        res, pred = self.garbage_identify.garbage_run(self.img)
        assert isinstance(res, np.ndarray)
        assert isinstance(pred, dict)

    # 测试较大index时，garbage_run函数输出类型
    def test_garbage_run_garbage_index(self):
        res, pred = self.garbage_identify.garbage_run(self.img, 6)
        assert isinstance(res, np.ndarray)
        assert isinstance(pred, dict)

    # 测试绿色垃圾时，机械臂的运动
    def test_garbage_grap_green(self):
        res = self.garbage_identify.garbage_grap(
            {"Fish_bone": (-0.0075, 0.21773)}, [89, 134]
        )
        assert res is None

    # 测试红色垃圾时，机械臂的运动
    def test_garbage_grap_red(self):
        res = self.garbage_identify.garbage_grap(
            {"Syringe": (-0.0075, 0.21773)}, [89, 134]
        )
        assert res is None

    # 测试蓝色垃圾时，机械臂的运动
    def test_garbage_grap_blue(self):
        res = self.garbage_identify.garbage_grap(
            {"Newspaper": (-0.0075, 0.21773)}, [89, 134]
        )
        assert res is None

    # 测试灰色垃圾时，机械臂的运动
    def test_garbage_grap_gray(self):
        res = self.garbage_identify.garbage_grap(
            {"Cigarette_butts": (-0.0075, 0.21773)}, [89, 134]
        )
        assert res is None
