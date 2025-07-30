#include "my_mindspore_ops_interface.hpp"
#include<iostream>
using namespace std;
int add_op(int i, int j) {
    printf("i: %d, j: %d\n", i, j);
    int ret = i + j;
    return ret;
}