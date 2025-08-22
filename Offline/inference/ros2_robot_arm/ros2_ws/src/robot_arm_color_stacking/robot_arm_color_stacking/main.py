#!/usr/bin/env python3
# coding: utf-8

import os
from time import sleep

import Arm_Lib
import cv2 as cv
from pathlib import Path
import numpy as np

from .utils.dofbot_config import ArmCalibration, read_XYT
from .utils.stacking_target import stacking_GetTarget


def main(args=None):
    # 创建获取目标实例
    target = stacking_GetTarget()
    # 创建相机标定实例
    calibration = ArmCalibration()
    # 初始化一些参数
    dp = []
    xy = [90, 135]
    msg = {}
    WARMUP_BUFFER = 25

    # 后续作为ROS参数
    DP_PRINT = False

    FILE = Path(__file__).resolve()
    lib_root = FILE.parents[0]
    lib_site_pkg = os.path.dirname(lib_root)
    lib_python = os.path.dirname(lib_site_pkg)
    lib_path = os.path.dirname(lib_python)
    shared_path = os.path.join(os.path.dirname(lib_path), "share")
    share_root = os.path.join(shared_path, "robot_arm_color_stacking")
    cfg_folder = os.path.join(share_root, "config")
    dp_cfg_path = os.path.join(cfg_folder, "dp.bin")

    # XYT参数路径
    # revise
    XYT_path = os.path.join(cfg_folder, "XYT_config.txt")

    try:
        xy, _ = read_XYT(XYT_path)
    except Exception:
        print("No XYT_config file!!!")

    print("Read xy is", xy)

    warm_up_count = 0
    last_num = 0
    last_count = 0

    # 创建机械臂驱动实例
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
            print("dp dtype:", dp.dtype)
            print(dp.shape)
            print(dp)
        dp = dp.reshape(4, 2)

        img = calibration.perspective_transform(dp, img)

        img, msg = target.select_color(img)

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
                target.target_run(msg, xy)
                warm_up_count = 0

    cv.destroyAllWindows()
    capture.release()


if __name__ == "__main__":
    main()
