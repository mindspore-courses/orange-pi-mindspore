import os
from glob import glob

from setuptools import setup


package_name = "robot_arm_color_stacking"
pkg_utils_path = package_name + ".utils"
pkg_config_path = package_name + "/config/*"
pkg_model_path = package_name + "/model/*"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name, pkg_utils_path],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob(pkg_config_path)),
        (os.path.join("share", package_name, "model"), glob(pkg_model_path)),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="root",
    maintainer_email="root@todo.todo",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": ["color_stacking = robot_arm_color_stacking.main:main"],
    },
)
