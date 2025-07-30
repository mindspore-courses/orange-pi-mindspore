#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import sys
import stat
import os
import re
import json
import const_var


def write_options_to_file(file_name: str, options_str: str, \
    op_type: str, compute_unit: str, split_char: str):
    flags = os.O_WRONLY | os.O_CREAT
    modes = stat.S_IWUSR | stat.S_IRUSR
    try:
        with os.fdopen(os.open(file_name, flags, modes), 'a') as fd:
            fd.write(op_type + split_char + compute_unit + split_char + options_str + '\n')
    except Exception as err:
        print("write compile options config file failed")
        raise(err)


def gen_compile_options(compile_options_file: str, op_type: str, \
    compute_unit: str, compile_options: list):
    base_dir = os.path.dirname(compile_options_file)
    opc_config_file = os.path.join(base_dir, "custom_opc_options.ini")
    compile_opt = []
    opc_debug_config = []
    opc_tiling_keys = ""
    for opts in compile_options:
        if "oom" in opts:
            if opts == "--oom":
                opc_debug_config.append("oom")
            else:
                raise RuntimeError(f"Unknown oom option format {opts}")
        elif "--save-temp-files" in opts:
            opc_debug_config.append("dump_cce")
        elif "--tiling_key" in opts:
            keys = opts.strip().split('=')[1].split(',')
            keys_str = ";".join([key for key in keys])
            opc_tiling_keys = keys_str
        else:
            compile_opt.append(opts)
    if len(compile_opt) > 0:
        options_str = ';'.join([opt for opt in compile_opt])
        write_options_to_file(compile_options_file, options_str, op_type, compute_unit, ",")
    opc_config_str = ""
    if opc_debug_config:
        opc_config_str = "--op_debug_config=" + ';'.join([opt for opt in opc_debug_config]) 
    if len(opc_tiling_keys) > 0:
        if opc_config_str != "":
            opc_config_str += "@"
        opc_config_str += "--tiling_key=" + opc_tiling_keys

    if opc_config_str != "":
        write_options_to_file(opc_config_file, opc_config_str, op_type, compute_unit, "@") 


if __name__ == '__main__':
    if len(sys.argv) < 4:
        raise RuntimeError('arguments must greater than 4')
    compute_soc = ""
    comp_options = []
    for i in range(len(sys.argv) - 3):
        if sys.argv[i + 3].upper().startswith("ASCEND"):
            compute_soc += sys.argv[i + 3] + ";"
        else:
            comp_options.append(sys.argv[i + 3])
    if compute_soc != "":
        compute_soc = compute_soc[0:-1]
    gen_compile_options(sys.argv[1], sys.argv[2], compute_soc, comp_options)