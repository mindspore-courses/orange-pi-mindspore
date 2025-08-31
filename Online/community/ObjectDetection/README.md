# 目标检测（Object Detection）应用案例
本项目基于 MindYOLO 实现，使用轻量级 YOLOv5n 模型，在香橙派 AIpro 开发板上完成目标检测任务。该应用能够对输入图片中的目标进行识别与定位，并在 Notebook 中直接显示检测结果。

## 模型准备
模型名称：YOLOv5n

模型下载地址：https://download.mindspore.cn/toolkits/mindyolo/yolov5/yolov5n_300e_mAP273-9b16bd7b.ckpt

说明：YOLOv5n 模型体积约 7.2MB，远小于 4GB，满足任务要求。
## 框架准备
目标检测框架名称：MindYOLO

源码压缩包下载地址：https://github.com/mindspore-lab/mindyolo/archive/refs/heads/master.zip

说明：解压后目录命名为 mindyolo_local，并通过 pip install --no-deps -e mindyolo_local 安装。
## 测试图片
文件名：bus.jpg

下载地址：https://ultralytics.com/images/bus.jpg

说明：作为示例输入图像，模型会在图中检测出行人、车辆等目标。


