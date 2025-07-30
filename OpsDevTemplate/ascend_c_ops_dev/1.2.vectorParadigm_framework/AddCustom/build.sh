#!/bin/bash
if [ -z "$BASE_LIBS_PATH" ]; then 
  if [ -z "$ASCEND_HOME_PATH" ]; then 
    if [ -z "$ASCEND_AICPU_PATH" ]; then 
      echo "please set env."
      exit 1
    else
      export ASCEND_HOME_PATH=$ASCEND_AICPU_PATH
    fi
  else 
    export ASCEND_HOME_PATH=$ASCEND_HOME_PATH
  fi
else
  export ASCEND_HOME_PATH=$BASE_LIBS_PATH
fi
echo "using ASCEND_HOME_PATH: $ASCEND_HOME_PATH"
script_path=$(realpath $(dirname $0))


mkdir -p build_out
rm -rf build_out/*
cd build_out

opts=$(python3 $script_path/cmake/util/preset_parse.py $script_path/CMakePresets.json)
ENABLE_CROSS="-DENABLE_CROSS_COMPILE=True"
ENABLE_BINARY="-DENABLE_BINARY_PACKAGE=True"
cmake_version=$(cmake --version | grep "cmake version" | awk '{print $3}')

cmake_run_package()
{
  target=$1
  cmake --build . --target $target -j16
  if [ $? -ne 0 ]; then exit 1; fi

  if [ $target = "package" ]; then
    if test -d ./op_kernel/binary ; then
      ./cust*.run
      if [ $? -ne 0 ]; then exit 1; fi
      cmake --build . --target binary -j16
      if [ $? -ne 0 ]; then
          echo "[ERROR] Kernel compile failed, the run package will not be generated."
          rm -rf ./cust*.run && rm -rf ./cust*.run.json && exit 1;
      fi
      cmake --build . --target $target -j16
    fi
  fi
}

if [[ $opts =~ $ENABLE_CROSS ]] && [[ $opts =~ $ENABLE_BINARY ]]
then
  target=package
  if [ "$1"x != ""x ]; then target=$1; fi
  if [ "$cmake_version" \< "3.19.0" ] ; then
    cmake .. $opts -DENABLE_CROSS_COMPILE=0 -DASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}
  else
    cmake .. --preset=default -DENABLE_CROSS_COMPILE=0 -DASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}
  fi
  cmake_run_package $target
  cp -r kernel ../
  rm -rf *
  if [ "$cmake_version" \< "3.19.0" ] ; then
    cmake .. $opts -DASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}
  else
    cmake .. --preset=default -DASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}
  fi

  cmake --build . --target $target -j16
  if [ $? -ne 0 ]; then
      echo "[ERROR] Kernel compile failed, the run package will not be generated."
      rm -rf ./cust*.run && rm -rf ./cust*.run.json && exit 1;
  fi
  if [ $target = "package" ]; then
    if test -d ./op_kernel/binary ; then
      ./cust*.run
    fi
  fi
  rm -rf ../kernel

else
  target=package
  if [ "$1"x != ""x ]; then target=$1; fi
  if [ "$cmake_version" \< "3.19.0" ] ; then
    cmake .. $opts -DASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}
  else
      cmake .. --preset=default -DASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}
  fi
  cmake_run_package $target
fi


# for debug
# cd build_out
# make
# cpack
# verbose append -v
