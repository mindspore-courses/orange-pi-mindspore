#include <pybind11/pybind11.h>
#include "my_mindspore_ops_interface.hpp"
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(my_mindspore_ops, m) { // 最后被import my_mindspore_ops的so文件
    m.doc() = R"pbdoc(
        Pybind11 example plugin for my_MindSpore
        -----------------------

        .. currentmodule:: my_mindspore_ops

        .. autosummary::
           :toctree: _generate

           add_op
           subtract_op
    )pbdoc";
    m.def("add_op", &add_op, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");
// 还可以直接定义匿名函数
    m.def("subtract_op", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers
        
        Some other explanation about the subtract function.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
