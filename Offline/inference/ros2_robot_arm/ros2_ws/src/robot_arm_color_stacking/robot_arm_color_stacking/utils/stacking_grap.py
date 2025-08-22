#!/usr/bin/env python
# coding: utf-8

from time import sleep

import Arm_Lib


class stacking_grap:
    def __init__(self):
        # 设置移动状态
        self.move_status = True
        # 创建机械臂实例
        self.arm = Arm_Lib.Arm_Device()
        # 夹爪加紧角度
        # 超过130夹的过紧会有响声
        self.grap_joint = 130

    def move(self, joints, joints_down):
        """
        移动过程
        :param joints: 移动到物体位置的各关节角度
        :param joints_down: 机械臂堆叠各关节角度
        """
        # 架起角度
        joints_00 = [90, 80, 50, 50, 265, self.grap_joint]
        # 抬起
        joints_up = [135, 80, 50, 50, 265, 30]
        # 架起
        self.arm.Arm_serial_servo_write6_array(joints_00, 1000)
        sleep(1)
        # 松开夹爪
        self.arm.Arm_serial_servo_write(6, 0, 500)
        sleep(0.5)
        # 移动至物体位置
        self.arm.Arm_serial_servo_write6_array(joints, 1000)
        sleep(1)
        # 夹紧夹爪
        self.arm.Arm_serial_servo_write(6, self.grap_joint, 500)
        sleep(0.5)
        # 架起
        self.arm.Arm_serial_servo_write6_array(joints_00, 1000)
        sleep(1)
        # 移动至对应位置上方
        self.arm.Arm_serial_servo_write(1, joints_down[0], 1000)
        sleep(1)
        # 移动至目标位置
        self.arm.Arm_serial_servo_write6_array(joints_down, 1000)
        sleep(1.5)
        # 松开夹爪
        self.arm.Arm_serial_servo_write(6, 0, 500)
        sleep(0.5)
        # 抬起
        self.arm.Arm_serial_servo_write6_array(joints_up, 1000)
        sleep(1)

    def arm_run(self, move_num, joints):
        """
        机械臂移动函数
        :param move_num: 抓取次数
        :param joints: 反解求得的各关节角度
        """
        if move_num == "1" and self.move_status:
            # 此处设置,需执行完本次操作,才能向下运行
            # 将设置机械臂的状态,以保证运行完此过程在执行其它
            self.move_status = False
            # 获得目标关节角
            joints = [joints[0], joints[1], joints[2], joints[3], 265, 30]
            # 移动到目标前的姿态
            joints_down = [135, 50, 20, 60, 265, self.grap_joint]
            # 调用移动函数
            self.move(joints, joints_down)
            # 执行完此过程
            self.move_status = True
        if move_num == "2" and self.move_status:
            self.move_status = False
            joints = [joints[0], joints[1], joints[2], joints[3], 265, 30]
            joints_down = [135, 55, 38, 38, 265, self.grap_joint]
            self.move(joints, joints_down)
            self.move_status = True
        if move_num == "3" and self.move_status:
            self.move_status = False
            joints = [joints[0], joints[1], joints[2], joints[3], 265, 30]
            joints_down = [135, 60, 45, 30, 265, self.grap_joint]
            self.move(joints, joints_down)
            self.move_status = True
        if move_num == "4" and self.move_status:
            self.move_status = False
            joints = [joints[0], joints[1], joints[2], joints[3], 265, 30]
            joints_down = [135, 65, 55, 20, 265, self.grap_joint]
            self.move(joints, joints_down)
            self.move_status = True
