#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Created on Feb  28 20:56:45 2020
Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
"""

import argparse
import sys
import os
import json
import hashlib
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import const_var
import opdesc_parser
 
PYF_PATH = os.path.dirname(os.path.realpath(__file__))


class BinParamBuilder(opdesc_parser.OpDesc):
    def __init__(self: any, op_type: str):
        super().__init__(op_type)
        self.soc = ''
        self.out_path = ''
        self.tiling_keys = set()
        self.op_debug_config = ''

    def set_soc_version(self: any, soc: str):
        self.soc = soc

    def set_out_path(self: any, out_path: str):
        self.out_path = out_path

    def set_tiling_key(self: any, tiling_key_info: Set):
        if tiling_key_info:
            self.tiling_keys.update(tiling_key_info)

    def set_op_debug_config(self: any, op_debug_config: str):
        if op_debug_config:
            self.op_debug_config = op_debug_config

    def gen_input_json(self: any):
        key_map = {}
        count = len(self.input_dtype[0].split(','))
        required_parameters = set()
        index_value = -1

        for i in range(0, count):
            inputs = []
            outputs = []
            attrs = []
            required_parameter = []
            op_node = {}

            for idx in range(0, len(self.input_name)):
                idtypes = self.input_dtype[idx].split(',')
                ifmts = self.input_fmt[idx].split(',')
                itype = self.input_type[idx]
                para = {}
                para['name'] = self.input_name[idx][:-5]
                para['index'] = idx
                para['dtype'] = idtypes[i]
                para['format'] = ifmts[i]
                para['paramType'] = itype
                para['shape'] = [-2]
                para['format_match_mode'] = 'FormatAgnostic'

                input_parameter_key = (idtypes[i], ifmts[i])
                if itype == 'dynamic':
                    inputs.append([para])
                    required_parameter.append(input_parameter_key)
                elif itype == 'required':
                    inputs.append(para)
                    required_parameter.append(input_parameter_key)
                else:
                    inputs.append(para)

            for idx in range(0, len(self.output_name)):
                odtypes = self.output_dtype[idx].split(',')
                ofmts = self.output_fmt[idx].split(',')
                otype = self.output_type[idx]
                para = {}
                para['name'] = self.output_name[idx][:-5]
                para['index'] = idx
                para['dtype'] = odtypes[i]
                para['format'] = ofmts[i]
                para['paramType'] = otype
                para['shape'] = [-2]
                para['format_match_mode'] = 'FormatAgnostic'
                output_parameter_key = (odtypes[i], ofmts[i])
                if otype == 'dynamic':
                    outputs.append([para])
                    required_parameter.append(output_parameter_key)
                elif otype == 'required':
                    outputs.append(para)
                    required_parameter.append(output_parameter_key)
                else:
                    outputs.append(para)

            for attr in self.attr_list:
                att = {}
                att['name'] = attr
                atype = self.attr_val.get(attr).get('type').lower()
                att['dtype'] = atype
                att['value'] = const_var.ATTR_DEF_VAL.get(atype)
                attrs.append(att)

            required_parameter_tuple = tuple(required_parameter)
            if required_parameter_tuple in required_parameters:
                continue
            else:
                required_parameters.add(required_parameter_tuple)
                index_value +=1

            op_node['bin_filename'] = ''
            op_node['inputs'] = inputs
            op_node['outputs'] = outputs
            if len(attrs) > 0:
                op_node['attrs'] = attrs

            param = {}
            param['op_type'] = self.op_type
            param['op_list'] = [op_node]
            objstr = json.dumps(param, indent='  ')
            md5sum = hashlib.md5(objstr.encode('utf-8')).hexdigest()
            while key_map.get(md5sum) is not None:
                objstr += '1'
                md5sum = hashlib.md5(objstr.encode('utf-8')).hexdigest()
            key_map[md5sum] = md5sum
            bin_file = self.op_type + '_' + md5sum
            op_node['bin_filename'] = bin_file
            param_file = os.path.join(self.out_path, bin_file + '_param.json')
            param_file = os.path.realpath(param_file)
            with os.fdopen(os.open(param_file, const_var.WFLAGS, const_var.WMODES), 'w') as fd:
                json.dump(param, fd, indent='  ')
            self._write_build_cmd(param_file, bin_file, index_value)


    def _write_build_cmd(self: any, param_file: str, bin_file: str, index: int):
        hard_soc = const_var.conv_soc_ver(self.soc)
        if not hard_soc:
            hard_soc = self.soc.capitalize()
        name_com = [self.op_type, self.op_file, str(index)]
        compile_file = os.path.join(self.out_path, '-'.join(name_com) + '.sh')
        compile_file = os.path.realpath(compile_file)

        bin_cmd_str = 'res=$(opc $1 --main_func={fun} --input_param={param} --soc_version={soc} \
                --output=$2 --impl_mode={impl} --simplified_key_mode=0 --op_mode=dynamic '

        build_cmd_var = "#!/bin/bash\n"
        build_cmd_var += f'echo "[{hard_soc}] Generating {bin_file} ..."\n'
        plog_level = os.environ.get("ASCEND_GLOBAL_LOG_LEVEL")
        plog_stdout = os.environ.get("ASCEND_SLOG_PRINT_TO_STDOUT")
        if plog_level is None:
            build_cmd_var += const_var.SET_PLOG_LEVEL_ERROR
        if plog_stdout is None:
            build_cmd_var += const_var.SET_PLOG_STDOUT
        build_cmd_var += const_var.SRC_ENV
        build_cmd_var += bin_cmd_str.format(fun=self.op_intf, soc=hard_soc, param=param_file, 
                                           impl='high_performance,optional')
        enable_tiling_keys = False
        if self.tiling_keys:
            tiling_keys_list = sorted(list(self.tiling_keys))
            tiling_key_str = ','.join([str(_key) for _key in tiling_keys_list])
            build_cmd_var += f' --tiling_key="{tiling_key_str}"'
            enable_tiling_keys = True

        if self.op_debug_config:
            op_debug_str = ','.join([str(_key) for _key in list(self.op_debug_config)])
            build_cmd_var += f' --op_debug_config={op_debug_str}'

        build_cmd_var += ")\n"
        build_cmd_var += "\n"
        if enable_tiling_keys is False:
            build_cmd_var += "echo \"${res}\"\n"
            build_cmd_var += const_var.CHK_CMD.format(res_file=bin_file + '.json')
            build_cmd_var += const_var.CHK_CMD.format(res_file=bin_file + '.o')
        else:
            build_cmd_var += "if [ $? -eq 1 ]; then\n"
            build_cmd_var += "    if echo \"${res}\" | \
grep -q \"None of the given tiling keys are in the supported list\"; then\n"
            build_cmd_var += "        echo \"${res}\"\n"
            build_cmd_var += "    else\n"
            build_cmd_var += "        echo \"${res}\"\n"
            build_cmd_var += "        exit 1\n"
            build_cmd_var += "    fi\n"
            build_cmd_var += "else\n"
            build_cmd_var += "echo \"${res}\"\n"
            build_cmd_var += const_var.CHK_CMD.format(res_file=bin_file + '.json')
            build_cmd_var += const_var.CHK_CMD.format(res_file=bin_file + '.o')
            build_cmd_var += "fi\n"
        build_cmd_var += f'echo "[{hard_soc}] Generating {bin_file} Done"\n'

        with os.fdopen(os.open(compile_file, const_var.WFLAGS, const_var.WMODES), 'w') as fd:
            fd.write(build_cmd_var)


def get_tiling_keys(tiling_keys: str) -> Set:
    all_tiling_keys = set()
    if not tiling_keys:
        return all_tiling_keys

    tiling_key_list = tiling_keys.split(';')
    for tiling_key_value in tiling_key_list:
        pattern = r"(?<![^\s])(\d+)-(\d+)(?![^\s])"
        results = re.findall(pattern, tiling_key_value)
        if results:
            start, end = results[0]
            if int(start) > int(end):
                continue
            for i in range(int(start), int(end) + 1):
                all_tiling_keys.add(i)
        elif tiling_key_value.isdigit():
            all_tiling_keys.add(int(tiling_key_value))
    return all_tiling_keys


def trans_soc_verion(soc_ver: str):
    low_soc_ver = soc_ver.lower()
    if low_soc_ver not in opdesc_parser.SOC_TO_SHORT_SOC_MAP:
        return low_soc_ver
    return opdesc_parser.SOC_TO_SHORT_SOC_MAP[low_soc_ver]


def parse_op_debug_confg(opc_config_file: str, soc: str) -> Dict:
    tiling_key_info = defaultdict(set)
    op_debug_config = defaultdict(set)
    if not opc_config_file:
        return tiling_key_info, op_debug_config

    if not os.path.exists(opc_config_file):
        return tiling_key_info, op_debug_config

    with open(opc_config_file, 'r') as file:
        contents = file.readlines()

    for _content in contents:
        content = _content.strip()
        opc_configs = content.split('@')
        if len(opc_configs) < 3:
            continue

        op_type = opc_configs[0]
        if not op_type:
            continue

        compute_unit = opc_configs[1]
        if compute_unit:
            compute_unit_list = compute_unit.split(';')
            soc_lists = []
            for soc_ver in compute_unit_list:
                short_soc_ver = trans_soc_verion(soc_ver)
                soc_lists.append(short_soc_ver)
            if soc not in soc_lists:
                continue

        for options in opc_configs[2:]:
            if "--tiling_key" in options:
                format_tiling_keys = get_tiling_keys(options.split('=')[1])
                if format_tiling_keys:
                    tiling_key_info[op_type].update(format_tiling_keys)
            if "--op_debug_config" in options:
                format_debug_config = set(options.split("=")[1].split(';'))
                if format_debug_config:
                    op_debug_config[op_type].update(format_debug_config)

    return tiling_key_info, op_debug_config


def gen_bin_param_file(cfgfile: str, out_dir: str, soc: str,
                        opc_config_file: str = '', ops: list = None):
    if not os.path.exists(cfgfile):
        print(f'INFO: {cfgfile} does not exists in this project, skip generating compile commands.')
        return

    op_descs = opdesc_parser.get_op_desc(cfgfile, [], [], BinParamBuilder, ops)
    tiling_key_info, op_debug_config = parse_op_debug_confg(opc_config_file, soc)

    all_soc_key = "ALL"
    for op_desc in op_descs:
        op_desc.set_soc_version(soc)
        op_desc.set_out_path(out_dir)
        if op_desc.op_type in op_debug_config:
            op_desc.set_op_debug_config(op_debug_config[op_desc.op_type])
        if all_soc_key in op_debug_config:
            op_desc.set_op_debug_config(op_debug_config[all_soc_key])
        if op_desc.op_type in tiling_key_info:
            op_desc.set_tiling_key(tiling_key_info[op_desc.op_type])
        if all_soc_key in tiling_key_info:
            op_desc.set_tiling_key(tiling_key_info[all_soc_key])
        op_desc.gen_input_json()


def parse_args(argv):
    """Command line parameter parsing"""
    parser = argparse.ArgumentParser()
    parser.add_argument('argv', nargs='+')
    parser.add_argument('--opc-config-file', nargs='?', const='', default='')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args(sys.argv)
    if len(args.argv) <= 3:
        raise RuntimeError('arguments must greater than 3')
    gen_bin_param_file(args.argv[1],
                    args.argv[2],
                    args.argv[3],
                    opc_config_file=args.opc_config_file)
