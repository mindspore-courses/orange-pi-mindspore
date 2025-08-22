#!/usr/bin/env python
# coding: utf-8

import os
from time import sleep

import Arm_Lib
import cv2 as cv
from pathlib import Path
import numpy as np

from .utils.dofbot_config import ArmCalibration, read_XYT
from .utils.garbage_identify import GarbageIdentify


def main(args=None):
    # 创建获取目标实例
    target = GarbageIdentify()
    # 创建相机标定实例
    calibration = ArmCalibration()
    # 创建机械臂驱动实例
    arm = Arm_Lib.Arm_Device()
    # 初始化一些参数
    # 初始化标定方框边点
    dp = []
    # 初始化抓取信息
    msg = {}
    # 初始化1,2舵机角度值
    xy = [90, 135]
    # 是否打印透视变换参数
    DP_PRINT = False
    # 预热值
    WARMUP_BUFFER = 25

    FILE = Path(__file__).resolve()
    lib_root = FILE.parents[0]
    lib_site_pkg = os.path.dirname(lib_root)
    lib_python = os.path.dirname(lib_site_pkg)
    lib_path = os.path.dirname(lib_python)
    shared_path = os.path.join(os.path.dirname(lib_path), "share")
    share_root = os.path.join(shared_path, "dofbot_garbage_yolov5")
    cfg_folder = os.path.join(share_root, "config")
    dp_cfg_path = os.path.join(cfg_folder, "dp.bin")

    # XYT参数路径
    XYT_path = os.path.join(cfg_folder, "XYT_config.txt")
    try:
        xy, _ = read_XYT(XYT_path)
    except Exception:
        print("Read XYT_config Error !!!")
        return

    print("Read xy is", xy)

    warm_up_count = 0
    last_num = 0
    last_count = 0

    arm = Arm_Lib.Arm_Device()
    joints_0 = [xy[0], xy[1], 0, 0, 90, 30]
    joints_1 = [xy[0], xy[1], 50, 50, 90, 30]

    # 重置机械臂位置
    print("Start Reset Robot Arm Position, Please Wait..")
    arm.Arm_serial_servo_write6_array(joints_1, 1000)
    sleep(2)
    arm.Arm_serial_servo_write6_array(joints_0, 1000)
    sleep(2)
    print("Finish Robot Arm Position Reset!")

    # 打开摄像头
    capture = cv.VideoCapture(0)
    # 当摄像头正常打开的情况下循环执行
    while capture.isOpened():
        # 读取相机的每一帧
        ret, img = capture.read()
        print("read image from camera successfully:", ret)
        # 统一图像大小
        img = cv.resize(img, (640, 480))

        dp = np.fromfile(dp_cfg_path, dtype=np.int32)
        if DP_PRINT:
            print("dp has dtype:", dp.dtype)
            print("dp has shape:", dp.shape)
        dp = dp.reshape(4, 2)
        if DP_PRINT:
            print("After reshape, dp has shape:", dp.shape)
            print("dp now is:", dp)

        img = calibration.perspective_transform(dp, img)

        img, msg = target.garbage_run(img)
        print("Model is warming up at stage:", warm_up_count)
        if warm_up_count != 0 and last_num == warm_up_count:
            last_count += 1
            if last_count > 5:
                warm_up_count = 0
                last_count = 0
        last_num = warm_up_count

        if len(msg) != 0:
            warm_up_count += 1
            if warm_up_count > WARMUP_BUFFER:
                target.garbage_grap(msg, xy)
                warm_up_count = 0
    return


if __name__ == "__main__":
    main()
