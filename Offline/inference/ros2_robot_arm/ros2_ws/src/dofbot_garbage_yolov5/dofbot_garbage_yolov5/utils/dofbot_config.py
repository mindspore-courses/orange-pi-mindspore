# !/usr/bin/env python
# coding: utf-8

import Arm_Lib
import cv2 as cv
import numpy as np


class ArmCalibration:
    def __init__(self):
        # 初始化图像
        self.image = None
        self.threshold_num = 130
        # 机械臂识别位置调节
        self.xy = [90, 135]
        # 创建机械臂驱动实例
        self.arm = Arm_Lib.Arm_Device()
        self.data_collect = True
        self.use_other_cam = False

    def calibration_map(self, image, xy=None, threshold_num=130):
        """
        放置方块区域检测函数
        :param image:输入图像
        :return:轮廓区域边点,处理后的图像
        """
        if xy != None:
            self.xy = xy
        # 机械臂初始位置角度
        joints_init = [self.xy[0], self.xy[1], 0, 0, 90, 0] 
        # 将机械臂移动到标定方框的状态
        self.arm.Arm_serial_servo_write6_array(joints_init, 500)
        self.image = image
        self.threshold_num = threshold_num
        # 创建边点容器
        dp = []
        h, w = self.image.shape[:2]
        # 获取轮廓点集(坐标)
        contours = self.morphological_processing()
        # 遍历点集
        for i, c in enumerate(contours):
            # 计算轮廓区域。
            area = cv.contourArea(c)

            low_bound = None
            high_bound = None
            if self.use_other_cam:
                low_bound = h * w / 2 - 40000
            else:
                low_bound = h * w / 2
            high_bound = h * w

            # 设置轮廓区域范围
            if low_bound < area < high_bound:
                # 计算多边形的矩
                mm = cv.moments(c)
                if mm["m00"] == 0:
                    continue
                cx = mm["m10"] / mm["m00"]
                cy = mm["m01"] / mm["m00"]
                # 绘制轮廓区域
                cv.drawContours(self.image, contours, i, (255, 255, 0), 2)
                # 获取轮廓区域边点
                dp = np.squeeze(cv.approxPolyDP(c, 100, True))
                # 绘制中心
                if not self.data_collect:
                    cv.circle(
                        self.image, (np.int_(cx), np.int_(cy)), 5, (0, 0, 255), -1
                    )
        return dp, self.image

    def morphological_processing(self):
        """
        形态学及去噪处理,并获取轮廓点集
        """
        # 将图像转为灰度图
        gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        # 使用高斯滤镜模糊图像。
        gray = cv.GaussianBlur(gray, (5, 5), 1)
        # 图像二值化操作
        _, threshold = cv.threshold(gray, self.threshold_num, 255, cv.THRESH_BINARY)
        # 获取不同形状的结构元素
        kernel = np.ones((3, 3), np.uint8)
        # 形态学开操作
        if self.use_other_cam:
            blur = cv.morphologyEx(threshold, cv.MORPH_OPEN, kernel, iterations=8)
        else:
            blur = cv.morphologyEx(threshold, cv.MORPH_OPEN, kernel, iterations=4)
        # 提取模式
        mode = cv.RETR_EXTERNAL
        # 提取方法
        method = cv.CHAIN_APPROX_NONE
        # 获取轮廓点集(坐标) python2和python3在此处略有不同
        # 层级关系 参数一：输入的二值图，参数二：提取模式，参数三：提取方法。
        contours, _ = cv.findContours(blur, mode, method)
        return contours

    def perspective_transform(self, dp, image):
        """
        透视变换
        :param dp: 方框边点(左上,左下,右下,右上)
        :param image: 原始图像
        :return: 透视变换后图像
        """
        if len(dp) != 4:
            return
        upper_left = []
        lower_left = []
        lower_right = []
        upper_right = []
        for i in range(len(dp)):
            if dp[i][0] < 320 and dp[i][1] < 240:
                upper_left = dp[i]
            if dp[i][0] < 320 and dp[i][1] > 240:
                lower_left = dp[i]
            if dp[i][0] > 320 and dp[i][1] > 240:
                lower_right = dp[i]
            if dp[i][0] > 320 and dp[i][1] < 240:
                upper_right = dp[i]
        # 原图中的四个顶点
        pts1 = np.float32([upper_left, lower_left, lower_right, upper_right])
        # 变换后的四个顶点
        pts2 = np.float32([[0, 0], [0, 480], [640, 480], [640, 0]])
        # 根据四对对应点计算透视变换。
        M = cv.getPerspectiveTransform(pts1, pts2)
        # 将透视变换应用于图像。
        Transform_img = cv.warpPerspective(image, M, (640, 480))
        return Transform_img


def write_XYT(wf_path, xy, thresh):
    with open(wf_path, "w") as wf:
        str1 = "x" + "=" + str(xy[0])
        str2 = "y" + "=" + str(xy[1])
        str3 = "thresh" + "=" + str(thresh)
        wf_str = str1 + "\n" + str2 + "\n" + str3
        wf.write(wf_str)
        wf.flush()


def read_XYT(rf_path):
    dict = {}
    rf = open(rf_path, "r+")
    for line in rf.readlines():
        index = line.find("=")
        dict[line[:index]] = line[index + 1 :]
    xy = [int(dict["x"]), int(dict["y"])]
    thresh = int(dict["thresh"])
    rf.flush()
    return xy, thresh
