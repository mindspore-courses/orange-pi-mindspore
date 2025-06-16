#!/bin/bash
CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)
cd $CURRENT_DIR

BUILD_TYPE="Debug"
INSTALL_PREFIX="${CURRENT_DIR}/out"

SHORT=r:,v:,i:,b:,p:,
LONG=run-mode:,soc-version:,install-path:,build-type:,install-prefix:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"
SOC_VERSION="Ascend310P3"

while :; do
    case "$1" in
    -r | --run-mode)
        RUN_MODE="$2"
        shift 2
        ;;
    -v | --soc-version)
        SOC_VERSION="$2"
        shift 2
        ;;
    -i | --install-path)
        ASCEND_INSTALL_PATH="$2"
        shift 2
        ;;
    -b | --build-type)
        BUILD_TYPE="$2"
        shift 2
        ;;
    -p | --install-prefix)
        INSTALL_PREFIX="$2"
        shift 2
        ;;
    --)
        shift
        break
        ;;
    *)
        echo "[ERROR] Unexpected option: $1"
        break
        ;;
    esac
done

RUN_MODE_LIST="cpu sim npu"
if [[ " $RUN_MODE_LIST " != *" $RUN_MODE "* ]]; then
    echo "ERROR: RUN_MODE error, This sample only support specify cpu, sim or npu!"
    exit -1
fi

VERSION_LIST="Ascend910A Ascend910B Ascend310B1 Ascend310B2 Ascend310B3 Ascend310B4 Ascend310P1 Ascend310P3 Ascend910B1 Ascend910B2 Ascend910B3 Ascend910B4"
if [[ " $VERSION_LIST " != *" $SOC_VERSION "* ]]; then
    echo "ERROR: SOC_VERSION should be in [$VERSION_LIST]"
    exit -1
fi

if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi

export ASCEND_TOOLKIT_HOME=${_ASCEND_INSTALL_PATH}
export ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}
echo "Current compile soc version is ${SOC_VERSION}"
source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash
if [ "${RUN_MODE}" = "sim" ]; then
    # in case of running op in simulator, use stub .so instead
    export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
    if [ ! $CAMODEL_LOG_PATH ]; then
        export CAMODEL_LOG_PATH=$(pwd)/sim_log
    fi
    if [ -d "$CAMODEL_LOG_PATH" ]; then
        rm -rf $CAMODEL_LOG_PATH
    fi
    mkdir -p $CAMODEL_LOG_PATH
elif [ "${RUN_MODE}" = "cpu" ]; then
    export LD_LIBRARY_PATH=${_ASCEND_INSTALL_PATH}/tools/tikicpulib/lib:${_ASCEND_INSTALL_PATH}/tools/tikicpulib/lib/${SOC_VERSION}:${_ASCEND_INSTALL_PATH}/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
fi

set -e
rm -rf build out
mkdir -p build
cmake -B build \
    -DRUN_MODE=${RUN_MODE} \
    -DSOC_VERSION=${SOC_VERSION} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
    -DASCEND_CANN_PACKAGE_PATH=${_ASCEND_INSTALL_PATH}
cmake --build build -j
cmake --install build

rm -f ascendc_kernels_bbit
cp ./out/bin/ascendc_kernels_bbit ./
rm -rf input output
mkdir -p input output
python3 scripts/gen_data.py
(
    export LD_LIBRARY_PATH=$(pwd)/out/lib:$(pwd)/out/lib64:${_ASCEND_INSTALL_PATH}/lib64:$LD_LIBRARY_PATH
    if [[ "$RUN_WITH_TOOLCHAIN" -eq 1 ]]; then
        if [ "${RUN_MODE}" = "npu" ]; then
            msprof op --application=./ascendc_kernels_bbit
        elif [ "${RUN_MODE}" = "sim" ]; then
            msprof op simulator --application=./ascendc_kernels_bbit
        elif [ "${RUN_MODE}" = "cpu" ]; then
            ./ascendc_kernels_bbit
        fi
    else
        ./ascendc_kernels_bbit
    fi
)
md5sum output/*.bin
python3 scripts/verify_result.py output/output_z.bin output/golden.bin
