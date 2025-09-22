#!/usr/bin/env python3
# coding: utf-8
import sys
import os
from time import sleep

import torch
import rclpy
import Arm_Lib
import cv2 as cv
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from pathlib import Path
from ais_bench.infer.interface import InferSession

from dofbot_info.srv import Kinemarics
from .stacking_grap import stacking_grap
from .npu_utils import (
    get_labels_from_txt,
    infer_image,
    xyxy2xywh,
)


rclpy.init(args=sys.argv)


class stacking_GetTarget:
    def __init__(self, test_mode=False):
        self.cfg = {
            "conf_thres": 0.85,
            "iou_thres": 0.8,
            "input_shape": [640, 640],
        }

        FILE = Path(__file__).resolve()
        lib_root = os.path.dirname(FILE.parents[0])
        lib_site_pkg = os.path.dirname(lib_root)
        lib_python = os.path.dirname(lib_site_pkg)
        lib_path = os.path.dirname(lib_python)
        shared_path = os.path.join(os.path.dirname(lib_path), "share")
        share_root = os.path.join(shared_path, "robot_arm_color_stacking")
        model_folder = os.path.join(share_root, "model")
        model_path = os.path.join(model_folder, "yolov5s_bs1.om")
        label_path = os.path.join(model_folder, "coco_names.txt")
        cfg_folder = os.path.join(share_root, "config")

        if test_mode:
            lib_root = os.path.dirname(FILE.parents[0])
            model_path = os.path.join(lib_root, "model", "yolov5s_bs1.om")
            label_path = os.path.join(lib_root, "model", "coco_names.txt")
            cfg_folder = os.path.join(lib_root, "config")

        self.offset_cfg_path = os.path.join(cfg_folder, "offset.txt")
        self.model = InferSession(0, model_path)
        self.labels_dict = get_labels_from_txt(label_path)
        self.test_mode = test_mode

        self.image = None
        self.color_name = None
        self.color_status = True
        # 机械臂识别位置调节
        self.xy = [90, 135]
        # 创建机械臂实例
        self.arm = Arm_Lib.Arm_Device()
        # 创建抓取实例
        self.grap = stacking_grap()
        # ROS节点初始化
        self.node = rclpy.create_node("dofbot_stacking")
        self.node_pub = rclpy.create_node("dofbot_img_node")
        # 创建获取反解结果的客户端
        self.client = self.node.create_client(Kinemarics, "trial_service")
        # 用于预热模型
        self.garbage_index = 0
        # 创建ROS发布摄像头图像信息
        self.image_pub = self.node_pub.create_publisher(Image, "cam_data", 10)
        self.bridge = CvBridge()

        self.offset = -1
        self.x_offset = -1
        with open(self.offset_cfg_path, "r") as f:
            self.offset = float(f.readline())
            self.x_offset = float(f.readline())
            print("y_offset is", self.offset)
            print("x_offset is", self.x_offset)
        print("finish init..")

    def target_run(self, msg, xy=None):
        """
        抓取函数
        :param msg: (颜色,位置)
        """
        if xy != None:
            self.xy = xy
        num = 1
        move_status = 0

        for i in msg.values():
            if i != None:
                move_status = 1

        if move_status == 1:
            self.arm.Arm_Buzzer_On(1)
            sleep(0.5)

        msg_list = sorted(list(msg.items()), key=lambda x: x[1][1])
        for name, pos in msg_list:
            print("pos : ", pos)
            print("name : ", name)
            try:
                joints = self.server_joint(pos)
                self.grap.arm_run(str(num), joints)
                num += 1
            except Exception:
                print("sqaure_pos empty")

        # 返回至中心位置
        self.arm.Arm_serial_servo_write(1, 90, 1000)
        sleep(1)
        # 初始位置
        joints_0 = [self.xy[0], self.xy[1], 0, 0, 90, 30]
        # 移动至初始位置
        self.arm.Arm_serial_servo_write6_array(joints_0, 1000)
        sleep(1)

    def select_color(self, image, garbage_index=0):
        """
        选择识别颜色
        :param image:输入图像
        :return: 输出处理后的图像,(颜色,位置)
        """

        if self.test_mode:
            self.garbage_index = garbage_index

        # 规范输入图像大小
        self.image = cv.resize(image, (640, 480))
        txt0 = "Model-Loading..."
        msg = {}
        # 模型预热 # revise
        if self.garbage_index < 5:
            self.garbage_index += 1
            return self.image, msg
        if self.garbage_index >= 5:
            # 创建消息容器
            try:
                # 获取识别消息
                msg = self.get_pos()
            except Exception:
                print("get_pos NoneType")
            return self.image, msg

        return self.image, msg

    def get_pos(self):
        """
        获取识别信息
        :return: 名称,位置
        """
        # 复制原始图像,避免处理过程中干扰
        img = self.image.copy()

        pred, names, drawed_res = infer_image(
            img, self.model, self.labels_dict, self.cfg
        )
        self.frame = drawed_res
        if len(pred) >= 1:
            cv.putText(
                self.frame,
                "Fix Infer Result, Waiting for Robot Arm Finish..",
                (30, 50),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        data = self.bridge.cv2_to_imgmsg(drawed_res, encoding="bgr8")
        self.image_pub.publish(data)

        pred = [pred]
        msg = {}
        gn = torch.tensor([640, 480, 640, 480])
        if pred:
            # Process detections
            block_ctr = 0
            # detections per image
            for i, det in enumerate(pred):
                for *xyxy, conf, cls in reversed(det):
                    prediction_status = True
                    # normalized xywh
                    xywh = (
                        (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                        .view(-1)
                        .tolist()
                    )
                    label = "%s %.2f" % (names[int(cls)], conf)
                    # get name
                    name = names[int(cls)]
                    name = name + str(block_ctr)

                    if prediction_status:
                        point_x = np.int_(xywh[0] * 640)
                        point_y = np.int_(xywh[1] * 480)
                        cv.circle(self.image, (point_x, point_y), 5, (0, 0, 255), -1)

                        # 计算方块在图像中的位置
                        (a, b) = (
                            round(((point_x - 320) / 4000), 5),
                            round(((480 - point_y) / 3000) * 0.8 + 0.19, 5),
                        )
                        msg[name] = (a, b)
                        block_ctr += 1
        return msg

    def server_joint(self, posxy):
        """
        发布位置请求,获取关节旋转角度
        :param posxy: 位置点x,y坐标
        :return: 每个关节旋转角度
        """
        # 等待server端启动
        self.client.wait_for_service()
        # 创建消息包
        request = Kinemarics.Request()
        request.tar_x = posxy[0] + self.x_offset
        request.tar_y = posxy[1] + self.offset
        request.kin_name = "ik"
        try:
            self.future = self.client.call_async(request)
            rclpy.spin_until_future_complete(self.node, self.future)
            response = self.future.result()
            if response:
                # 获得反解响应结果
                joints = [0.0, 0.0, 0.0, 0.0, 0.0]
                joints[0] = response.joint1
                joints[1] = response.joint2
                joints[2] = response.joint3
                joints[3] = response.joint4
                joints[4] = response.joint5
                # 当逆解越界,出现负值时,适当调节.
                if joints[2] < 0:
                    joints[1] += joints[2] * 3 / 5
                    joints[3] += joints[2] * 3 / 5
                    joints[2] = 0
                return joints
        except Exception:
            print("arg error")
