#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import sys, os

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__), './build'))  
import add_custom

torch.npu.config.allow_internal_format = False


class TestCustomAdd(TestCase):

    def test_add_custom_ops(self):
        length = [8, 2048]
        x = torch.rand(length, device='cpu', dtype=torch.float16)
        y = torch.rand(length, device='cpu', dtype=torch.float16)

        x_npu = x.npu()
        y_npu = y.npu()
        output = add_custom.run_add_custom(x_npu, y_npu)
        print("npuout customize add=", output)
        cpuout = torch.add(x, y)
        print("cpuout normalize add=", cpuout)

        self.assertRtolEqual(output, cpuout)


if __name__ == "__main__":
    run_tests()
