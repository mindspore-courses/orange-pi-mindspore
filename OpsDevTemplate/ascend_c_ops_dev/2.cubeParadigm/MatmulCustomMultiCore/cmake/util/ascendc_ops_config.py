#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Created on Feb  28 20:56:45 2020
Copyright (c) Huawei Technologies Co., Ltd. 2020-2024. All rights reserved.
"""

import os
import glob
import json
import sys
import argparse
import const_var


BINARY_INFO_CONFIG_JSON = "binary_info_config.json"


def load_json(json_file: str):
    with open(json_file, encoding='utf-8') as file:
        json_content = json.load(file)
    return json_content


def get_specified_suffix_file(root_dir, suffix):
    specified_suffix = os.path.join(root_dir, '**/*.{}'.format(suffix))
    all_suffix_files = glob.glob(specified_suffix, recursive=True)
    return sorted(all_suffix_files)


def add_dict_key(dict_to_add, key, value):
    if value is None:
        return
    dict_to_add[key] = value


def correct_format_mode(format_mode):
    if format_mode == 'FormatDefault':
        return 'nd_agnostic'
    if format_mode == 'FormatAgnostic':
        return 'static_nd_agnostic'
    if format_mode == 'FormatFixed':
        return 'normal'
    return format_mode


def get_input_or_output_config(in_or_out):
    param_dict = {}
    name = in_or_out.get('name')
    index = in_or_out.get('index')
    param_type = in_or_out.get('paramType')

    format_match_mode = in_or_out.get('format_match_mode')
    format_mode = correct_format_mode(format_match_mode)

    dtype_mode = in_or_out.get('dtype_match_mode')
    if dtype_mode == 'DtypeByte':
        dtype_mode = 'bit'

    add_dict_key(param_dict, 'name', name)
    add_dict_key(param_dict, 'index', index)
    add_dict_key(param_dict, 'paramType', param_type)
    add_dict_key(param_dict, 'dtypeMode', dtype_mode)
    add_dict_key(param_dict, 'formatMode', format_mode)
    return param_dict


def get_inputs_or_outputs_config(inputs_or_outputs):
    if inputs_or_outputs is None:
        return None
    inputs_or_outputs_list = []

    for in_or_out in inputs_or_outputs:
        if isinstance(in_or_out, dict):
            dict_param_config = get_input_or_output_config(in_or_out)
            inputs_or_outputs_list.append(dict_param_config)
        elif isinstance(in_or_out, list):
            param_info = in_or_out[0]
            list_param_config = get_input_or_output_config(param_info)
            tmp_list = [list_param_config]
            inputs_or_outputs_list.append(tmp_list)
    return inputs_or_outputs_list


def gen_attrs_config(attrs):
    attrs_list = []
    for attr in attrs:
        attrs_dict = {}
        name = attr.get('name')
        mode = attr.get('mode')
        add_dict_key(attrs_dict, 'name', name)
        add_dict_key(attrs_dict, 'mode', mode)
        attrs_list.append(attrs_dict)
    return attrs_list


def get_params_config(support_info):
    params_dict = {}

    inputs = support_info.get('inputs')
    inputs_list = get_inputs_or_outputs_config(inputs)
    params_dict['inputs'] = inputs_list

    outputs = support_info.get('outputs')
    outputs_list = get_inputs_or_outputs_config(outputs)
    params_dict['outputs'] = outputs_list

    attrs = support_info.get('attrs')
    if attrs is not None:
        attrs_list = gen_attrs_config(attrs)
        params_dict['attrs'] = attrs_list

    return params_dict


def add_simplified_config(op_type, support_info, core_type, task_ration, objfile, config):
    simplified_key = support_info.get('simplifiedKey')

    json_path = objfile.split('.')[0] + '.json'

    simple_cfg = config.get(BINARY_INFO_CONFIG_JSON)
    op_cfg = simple_cfg.get(op_type)
    if not op_cfg:
        op_cfg = {'dynamicRankSupport': True}

        simplified_key_mode = support_info.get('simplifiedKeyMode')
        add_dict_key(op_cfg, 'simplifiedKeyMode', simplified_key_mode)

        optional_input_mode = support_info.get('optionalInputMode')
        add_dict_key(op_cfg, 'optionalInputMode', optional_input_mode)

        params_info = get_params_config(support_info)
        op_cfg['params'] = params_info
        op_cfg['binaryList'] = []
        simple_cfg[op_type] = op_cfg

    bin_list = op_cfg.get('binaryList')
    if core_type == 0 and task_ration == "tilingKey":
        bin_list.append({'coreType': core_type, 'simplifiedKey': simplified_key,
                         'multiKernelType': 1, 'binPath': objfile, 'jsonPath': json_path})
    else:
        bin_list.append({'coreType': core_type, 'simplifiedKey': simplified_key,
                         'binPath': objfile, 'jsonPath': json_path})


def add_op_config(op_file, bin_info, config):
    op_cfg = config.get(op_file)
    if not op_cfg:
        op_cfg = {'binList': []}
        config[op_file] = op_cfg
    op_cfg.get('binList').append(bin_info)


def gen_ops_config(json_file, soc, config):
    core_type_map = {'MIX': 0, 'AiCore': 1, 'VectorCore': 2, 'MIX_AICORE': 3, 'MIX_VECTOR_CORE': 4}
    contents = load_json(json_file)
    if ('binFileName' not in contents) or ('supportInfo' not in contents):
        return
    json_base_name = os.path.basename(json_file)
    op_dir = os.path.basename(os.path.dirname(json_file))

    support_info = contents.get('supportInfo')
    bin_name = contents.get('binFileName')
    bin_suffix = contents.get('binFileSuffix')
    core_type = core_type_map.get(contents.get("coreType"))
    task_ration = contents.get("taskRation")
    bin_file_name = bin_name + bin_suffix
    op_type = bin_name.split('_')[0]
    op_file = op_dir + '.json'
    bin_info = {}

    add_dict_key(bin_info, 'implMode', support_info.get('implMode'))
    add_dict_key(bin_info, 'int64Mode', support_info.get('int64Mode'))
    add_dict_key(bin_info, 'simplifiedKeyMode', support_info.get('simplifiedKeyMode'))

    simplified_key = support_info.get('simplifiedKey')
    if simplified_key is not None:
        bin_info['simplifiedKey'] = simplified_key
        obj_file = os.path.join(soc, op_dir, bin_file_name)
        add_simplified_config(op_type, support_info, core_type, task_ration, obj_file, config)

    add_dict_key(bin_info, 'dynamicParamMode', support_info.get('dynamicParamMode'))
    bin_info['staticKey'] = support_info.get('staticKey')
    bin_info['inputs'] = support_info.get('inputs')
    bin_info['outputs'] = support_info.get('outputs')
    if support_info.get('attrs'):
        bin_info['attrs'] = support_info.get('attrs')

    add_dict_key(bin_info, 'opMode', support_info.get('opMode'))
    add_dict_key(bin_info, 'optionalInputMode', support_info.get('optionalInputMode'))
    add_dict_key(bin_info, 'deterministic', support_info.get('deterministic'))

    bin_info['binInfo'] = {'jsonFilePath': os.path.join(soc, op_dir, json_base_name)}
    add_op_config(op_file, bin_info, config)


def check_single_op_is_void(root_dir):
    for root, dirs, _ in os.walk(root_dir):
        for sub_dir in dirs:
            dir_path = os.path.join(root, sub_dir)
            if len(os.listdir(dir_path)) == 0:
                print(f"[ERROR] op {sub_dir}: not any obj compile success")
                sys.exit(1)


def gen_all_config(root_dir, soc, out_dir, skip_binary_info_config):
    suffix = 'json'
    config = {BINARY_INFO_CONFIG_JSON: {}}
    check_single_op_is_void(root_dir)
    all_json_files = get_specified_suffix_file(root_dir, suffix)

    for _json in all_json_files:
        gen_ops_config(_json, soc, config)
        file_path = soc + _json.split(soc)[1]
        with open(_json, "r+") as f:
            data = json.load(f)
            data["filePath"] = file_path
            f.seek(0)
            json.dump(data, f, indent=" ")
            f.truncate()

    for cfg_key in config.keys():
        if skip_binary_info_config and cfg_key == BINARY_INFO_CONFIG_JSON:
            continue
        cfg_file = os.path.join(out_dir, cfg_key)
        with os.fdopen(os.open(cfg_file, const_var.WFLAGS, const_var.WMODES), 'w') as fd:
            json.dump(config.get(cfg_key), fd, indent='  ')


# Parse multiple soc_versions ops in single path.
def gen_all_soc_config(all_path):
    soc_roots = glob.glob(os.path.join(all_path, "ascend*"))

    for soc_root in soc_roots:
        soc = os.path.basename(soc_root)
        gen_all_config(soc_root, soc, soc_root, True)
        cfg_files = glob.glob(os.path.join(soc_root, "*.json"))
        cfg_path = os.path.join(all_path, "config", soc)
        os.makedirs(cfg_path, exist_ok=True)
        for cfg_file in cfg_files:
            new_file = os.path.join(cfg_path, os.path.basename(cfg_file))
            os.rename(cfg_file, new_file)


def args_prase():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--path',
                        nargs='?',
                        required=True,
                        help='Parse the path of the json file.')

    parser.add_argument('-s',
                        '--soc',
                        nargs='?',
                        required=True,
                        help='Parse the soc_version of ops.')

    parser.add_argument('-o',
                        '--out',
                        nargs='?',
                        help='Output directory.')

    parser.add_argument('--skip-binary-info-config',
                        action='store_true',
                        help='binary_info_config.json file is not parsed.')

    return parser.parse_args()


def main():
    args = args_prase()
    if args.out is None:
        out_dir = args.path
    else:
        out_dir = args.out

    gen_all_config(args.path, args.soc, out_dir, args.skip_binary_info_config)


if __name__ == '__main__':
    main()
