import os

import numpy as np
import cv2
from ais_bench.infer.interface import InferSession
from pathlib import Path

from dofbot_garbage_yolov5.utils.npu_utils import (
    preprocess_image,
    get_labels_from_txt,
    infer_image,
    xyxy2xywh,
)


class TestNPUUtils:
    def setup_method(self):
        self.cfg = {
            "conf_thres": 0.7,
            "iou_thres": 0.7,
            "input_shape": [640, 640],
        }

        FILE = Path(__file__).resolve()
        pkg_root_dir = os.path.dirname(FILE.parents[0])
        model_dir = os.path.join(pkg_root_dir, "dofbot_garbage_yolov5", "model")
        model_path = os.path.join(model_dir, "yolov5s_bs1.om")
        test_dir = os.path.join(pkg_root_dir, "dofbot_garbage_yolov5", "test_imgs")
        self.model = InferSession(0, model_path)
        self.label_path = os.path.join(model_dir, "coco_names.txt")
        self.labels_dict = get_labels_from_txt(self.label_path)
        self.img_bgr = cv2.imread(os.path.join(test_dir, "2.jpg"))

    # 测试preprocess_image函数的输出类型
    def test_preprocess_image(self):
        img, scale_ratio, pad_size = preprocess_image(self.img_bgr, self.cfg)
        assert isinstance(img, np.ndarray)
        assert isinstance(scale_ratio, tuple)
        assert isinstance(pad_size, tuple)

    # 测试infer_image函数的输出类型
    def test_infer_image(self):
        pred_all, class_names, drawed_res = infer_image(
            self.img_bgr, self.model, self.labels_dict, self.cfg
        )
        assert isinstance(pred_all, np.ndarray)
        assert isinstance(class_names, dict)
        assert isinstance(drawed_res, np.ndarray)

    # 测试get_labels_from_txt函数的输出类型
    def test_get_labels_from_txt(self):
        labels_dict = get_labels_from_txt(self.label_path)
        assert isinstance(labels_dict, dict)

    # 测试xyxy2xywh函数的输出类型与输出值
    def test_xyxy2xywh(self):
        x = np.array([[100, 100, 200, 200], [300, 300, 400, 400]])
        y = xyxy2xywh(x)
        assert isinstance(y, np.ndarray)
        assert y[0][0] == 150
        assert y[0][1] == 150
        assert y[0][2] == 100
        assert y[0][3] == 100
        assert y[1][0] == 350
        assert y[1][1] == 350
        assert y[1][2] == 100
        assert y[1][3] == 100
