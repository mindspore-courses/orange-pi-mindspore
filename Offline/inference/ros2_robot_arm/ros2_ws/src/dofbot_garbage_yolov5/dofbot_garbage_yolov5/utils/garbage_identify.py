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

import mindspore_lite as mslite
from mindyolo.utils import logger

from dofbot_info.srv import Kinemarics

try:
    from garbage_grap import GarbageGrapMove
    from npu_utils import get_labels_from_txt, infer_image, xyxy2xywh
except:
    from .garbage_grap import GarbageGrapMove
    from .npu_utils import get_labels_from_txt, infer_image, xyxy2xywh

# 创建节点句柄/ROS节点初始化 - 放在外面的好处：可以解决多次初始化的问题（放在类中，只要新创建一个类对象，就会初始化一次）
rclpy.init(args=sys.argv)


class GarbageIdentify:
    def __init__(self, test_mode=False):
        self.cfg = {
            "conf_thres": 0.7,
            "iou_thres": 0.7,
            "input_shape": [640, 640],
        }

        # 创建ROS节点
        self.node = rclpy.create_node("dofbot_garbage")
        self.node_pub = rclpy.create_node("dofbot_img_node")

        FILE = Path(__file__).resolve()
        lib_root = os.path.dirname(FILE.parents[0])
        lib_site_pkg = os.path.dirname(lib_root)
        lib_python = os.path.dirname(lib_site_pkg)
        lib_path = os.path.dirname(lib_python)
        shared_path = os.path.join(os.path.dirname(lib_path), "share")
        share_root = os.path.join(shared_path, "dofbot_garbage_yolov5")
        model_folder = os.path.join(share_root, "model")
        model_path = os.path.join(model_folder, "yolov5s_lite_ros.mindir")
        label_path = os.path.join(model_folder, "coco_names.txt")
        cfg_folder = os.path.join(share_root, "config")
        offset_cfg_path = os.path.join(cfg_folder, "offset.txt")

        if test_mode:
            lib_root = os.path.dirname(FILE.parents[0])
            model_path = os.path.join(lib_root, "model", "yolov5s_lite_ros.mindir")
            label_path = os.path.join(lib_root, "model", "coco_names.txt")
            offset_cfg_path = os.path.join(lib_root, "config", "offset.txt")
        
        
        context = mslite.Context()
        context.target = ["Ascend"]
        self.model = mslite.Model()
        logger.info('mslite model init...')
        self.model.build_from_file(model_path,mslite.ModelType.MINDIR,context)
        
        self.labels_dict = get_labels_from_txt(label_path)
        self.WARMUP_INDEX = 5
        self.test_mode = test_mode

        # 初始化图像
        self.frame = None
        # 创建机械臂实例
        self.arm = Arm_Lib.Arm_Device()
        # 机械臂识别位置调节
        self.xy = [90, 130]
        self.garbage_index = 0
        # 创建垃圾识别抓取实例
        self.grap_move = GarbageGrapMove()
        # 创建用于调用的ROS服务的句柄
        self.client = self.node.create_client(Kinemarics, "trial_service")
        # 创建ROS发布摄像头图像信息
        self.image_pub = self.node_pub.create_publisher(Image, "cam_data", 10)
        self.bridge = CvBridge()

        self.offset = -1
        self.x_offset = -1
        with open(offset_cfg_path, "r") as f:
            self.offset = float(f.readline())
            self.x_offset = float(f.readline())
            print("y_offset is", self.offset)
            print("x_offset is", self.x_offset)
        print("finish init..")

    def garbage_grap(self, msg, xy=None):
        """
        执行抓取函数
        :param msg: {name:pos,...}
        :param xy: 机械臂初始位置 list, eg: [89, 134]
        """
        if xy != None:
            self.xy = xy
        if len(msg) != 0:
            self.arm.Arm_Buzzer_On(1)
            sleep(0.5)
        new_msg = sorted(list(msg.items()), key=lambda x: x[1][1])
        print("new msg is", new_msg)
        for elm in new_msg:
            try:
                name = elm[0]
                # 此处ROS反解通讯,获取各关节旋转角度
                joints = self.server_joint(msg[name])
                # 调取移动函数
                self.grap_move.arm_run(str(name), joints)
            except Exception:
                print("sqaure_pos empty")
        # 初始位置
        joints_0 = [self.xy[0], self.xy[1], 0, 0, 90, 30]
        print("back position is", joints_0)
        # 移动至初始位置
        self.arm.Arm_serial_servo_write6_array(joints_0, 1000)
        sleep(1)

    def garbage_run(self, image, garbage_index=None):
        """
        执行垃圾识别函数
        :param image: 原始图像
        :return: 识别后的图像,识别信息(name, msg)
        """
        if garbage_index:
            # for testing
            self.garbage_index = garbage_index

        # 规范输入图像大小
        self.frame = cv.resize(image, (640, 480))

        txt0 = "Model-Loading..."
        msg = {}
        # 模型预热
        if self.garbage_index < self.WARMUP_INDEX:
            cv.putText(
                self.frame, txt0, (190, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )
            self.garbage_index += 1
            return self.frame, msg
        if self.garbage_index >= self.WARMUP_INDEX:
            # 创建消息容器
            try:
                # 获取识别消息
                msg = self.get_pos()
                print("msg is:", msg)
            except Exception:
                print("get_pos NoneType")
            return self.frame, msg

    def get_pos(self):
        """
        获取识别信息
        :return: 名称,位置
        """
        # 复制原始图像,避免处理过程中干扰
        img = self.frame.copy()

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
        print(f'pred:{pred}')
        if pred:
            # detections per image
            for _, det in enumerate(pred):
                # Write results
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
                    if prediction_status:
                        point_x = np.int_(xywh[0] * 640)
                        point_y = np.int_(xywh[1] * 480)
                        cv.circle(self.frame, (point_x, point_y), 5, (0, 0, 255), -1)
                        # 计算方块在图像中的位置
                        (a, b) = (
                            round(((point_x - 320) / 4000), 5),
                            round(((480 - point_y) / 3000) * 0.8 + 0.19, 5),
                        )
                        msg[name] = (a, b)
        return msg

    def server_joint(self, posxy):
        """
        发布位置请求,获取关节旋转角度
        :param posxy: 位置点x,y坐标
        :return: 每个关节旋转角度
        """
        print("posxy is:", posxy)
        # 等待server端启动
        self.client.wait_for_service(timeout_sec=1.0)
        # 创建消息包
        request = Kinemarics.Request()
        request.tar_x = posxy[0] + self.x_offset
        # REVISE
        request.tar_y = posxy[1] + self.offset
        request.kin_name = "ik"
        try:
            self.future = self.client.call_async(request)
            rclpy.spin_until_future_complete(self.node, self.future)
            response = self.future.result()
            if response:
                # 获取反解的响应结果
                joints = [0, 0, 0, 0, 0]
                joints[0] = response.joint1
                joints[1] = response.joint2
                joints[2] = response.joint3
                joints[3] = response.joint4
                joints[4] = response.joint5
                # 角度调整
                if joints[2] < 0:
                    joints[1] += joints[2] / 2
                    joints[3] += joints[2] * 3 / 4
                    joints[2] = 0
                return joints
        except Exception:
            print("arg error")
