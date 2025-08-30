#ifndef ARM_MOVEIT_ARM_KINEMARICS_H
#define ARM_MOVEIT_ARM_KINEMARICS_H

#include <kdl/chain.hpp>
#include <kdl/tree.hpp>
#include <kdl/segment.hpp>
#include <kdl/frames_io.hpp>
#include <kdl/chainiksolverpos_lma.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <iostream>

using namespace KDL;
using namespace std;
class Dofbot{
public:
    /**
     * 正向运动学计算 根据旋转关节角获取当前位姿
     * @param urdf_file 模型文件路径
     * @param joints    当前关节角度
     * @param cartpos   当前末端位姿
     */
    bool dofbot_getFK(const char *urdf_file, vector<double> &joints, vector<double> &currentPos);

    /**
     * 逆运动学计算 获取到到目标点各关节需要转动的角度
     * @param urdf_file  模型文件路径
     * @param targetXYZ  目标位置
     * @param targetRPY  目标姿态
     * @param outjoints  目标点关节角度
     */
    bool dofbot_getIK(const char *urdf_file, vector<double> &targetXYZ, vector<double> &targetRPY, vector<double> &outjoints);
};

#endif //ARM_MOVEIT_ARM_KINEMARICS_H
