import os
import time
import logging
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from moviepy.editor import VideoFileClip
import mindspore
from mindspore import context, ops

from dct_generator import Generator
from download import download

def camera(network, args, watermark):
    """摄像头推理"""

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("无法打开摄像头")
        exit()

    while True:
        begin = time.time()
        ret, img = cap.read()
        if not ret:
            logging.error("无法读取视频流")
            time.sleep(10)
            continue

        img_h, img_w, _ = img.shape
        logging.debug("original image size:({}, {})".format(img_h, img_w))
        origin = img.copy()
        if args.speed_first == "yes":
            img = cv2.resize(img, (320, 240))
        else:
            img = cv2.resize(img, (img_w // 8 * 8, img_h // 8 * 8))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img[...,::-1] / 255.0 - 0.5) * 2 
        img = img.transpose(2, 0, 1)[np.newaxis,:].astype(np.float32)
        inp = mindspore.Tensor(img)
        xg = network(inp)[0]
        xg = (xg + 1) * 0.5
        xg = ops.clamp(xg * 255 + 0.5, 0, 255)
        xg = xg.permute(1, 2, 0).asnumpy()[...,::-1]
        xg = cv2.cvtColor(xg, cv2.COLOR_RGB2BGR)
        xg = xg.astype(np.uint8)
        xg = cv2.resize(xg, (origin.shape[1], origin.shape[0]))
        result = np.hstack((origin, xg))

        if watermark:
            text_left = "origin image:({}x{})".format(img_h, img_w)
            text_right = "processed image:({}x{})".format(img_h, img_w)
            add_watermark(result, 10, 20, text_left)
            add_watermark(result, 10 + img_w, 20, text_right)

        # 显示图像
        cv2.imshow('MindSpore Application [DCT-Net] on OrangePi', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出
            break
        end = time.time()
        logging.debug('process image cost time:{}'.format(end - begin))


def add_watermark(img, weight_bias, height_bias, text):
    """添加水印"""

    # 设置字体大小和颜色
    font_scale = 0.8  # 字体大小
    color = (255, 255, 255)  # 白色文字 (B, G, R)
    thickness = 2  # 文字厚度
    # 设置字体类型
    font = cv2.FONT_HERSHEY_SIMPLEX
    (_, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # 计算文本的起始位置
    x = weight_bias
    # Y轴的位置设置为文本高度 + 上边距
    y = text_height + height_bias  # 20 像素的上边距

    # 在图像上绘制文本
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness)


def proc_img(network, args):
    """单张图片推理"""
    img = cv2.imread(args.input_path)
    begin = time.time()
    img_h, img_w, _ = img.shape
    logging.info("original image size:({}, {})".format(img_h, img_w))
    img = cv2.resize(img, (img_w // 8 * 8, img_h // 8 * 8))
    img = (img[...,::-1] / 255.0 - 0.5) * 2
    img = img.transpose(2, 0, 1)[np.newaxis,:].astype(np.float32)
    inp = mindspore.Tensor(img)
    xg = network(inp)[0]
    xg = (xg + 1) * 0.5
    xg = ops.clamp(xg*255+0.5,0,255)
    xg = xg.permute(1,2,0).asnumpy()[...,::-1]
    cv2.imwrite(args.output_path, xg)
    end = time.time()
    logging.info('process image cost time:{}, output image path:{}'.format(end-begin, args.output_path))

def video_comparess(video_path, output_path):
    # 读取视频文件
    video = VideoFileClip(video_path)
    # 设置输出视频的比特率（可以根据需要调整）
    output_bitrate = "500k"  # 500 kbps
    # 写出压缩后的视频
    video.write_videofile(output_path, bitrate=output_bitrate)

def proc_video(network, args):
    """视频推理"""
    cap = cv2.VideoCapture(args.input_path)
    # 获取视频的帧率和尺寸
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数
    logging.debug("original video size:({}, {})".format(height, width))

    # 创建一个 VideoWriter 对象，用于写入视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4格式
    tmp_path = "tmp.mp4"
    out = cv2.VideoWriter(tmp_path, fourcc, fps, (width * 2, height))

    with tqdm(total=total_frames, desc='Processing Frames', unit='frame') as pbar:
        while cap.isOpened():
            ret, frame = cap.read()  # 读取一帧
            pbar.update(1)  # 更新进度条
            if not ret:  # 如果没有帧可读，退出循环
                break
            begin = time.time()

            origin = frame.copy()
            frame = cv2.resize(frame, (width // 8 * 8, height // 8 * 8))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = (frame[...,::-1] / 255.0 - 0.5) * 2
            frame = frame.transpose(2, 0, 1)[np.newaxis,:].astype(np.float32)
            inp = mindspore.Tensor(frame)
            xg = network(inp)[0]
            xg = (xg + 1) * 0.5
            xg = ops.clamp(xg * 255 + 0.5, 0, 255)
            xg = xg.permute(1, 2, 0).asnumpy()[...,::-1]
            xg = cv2.cvtColor(xg, cv2.COLOR_RGB2BGR)
            xg = xg.astype(np.uint8)
            xg = cv2.resize(xg, (width, height))
            result = np.hstack((origin, xg))
            out.write(result)
            end = time.time()
            logging.debug('process image cost time:{}'.format(end - begin))

    # 释放视频捕获对象
    cap.release()
    out.release()
    video_comparess(tmp_path, args.output_path)
    logging.info(f"处理完成，保存为 {args.output_path}")
    os.remove(tmp_path)


if __name__ == "__main__":
    # 创建解析器
    parser = argparse.ArgumentParser(description="DCT-Net Inference")
    # 添加参数
    parser.add_argument("--run_mode", type=str, default="GRAPH_MODE", help="运行模式：'GRAPH_MODE' or 'PYNATIVE'")
    parser.add_argument("--device_type", type=str, default="Ascend", help="设备类型：'CPU' or 'Ascend'")
    parser.add_argument("--device_id", type=int, default=0, help="设备ID")
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/dct-net.ckpt", help="模型文件路径")
    parser.add_argument("--camera", type=str, default="yes", help="yes:使用摄像头实时推理，no:使用本地图片推理")
    parser.add_argument("--speed_first", type=str, default="yes", help="yes:优先考虑速度，no:优先考虑质量")
    parser.add_argument("--input_path", type=str, default="./images/gdg.png", help="源图像/视频文件路径")
    parser.add_argument("--output_path", type=str, default="./images/output.png", help="输出图像/视频文件路径")
    parser.add_argument("--log_path", type=str, default="./run.log", help="日志文件路径")
    parser.add_argument("--log_level", type=int, default="2", help="日志级别：'1-DEBUG', '2-INFO', '3-WARNING', '4-ERROR', '5-CRITICAL'")
    
    # 解析参数
    args = parser.parse_args()

    # 配置日志记录
    logging.basicConfig(filename=args.log_path,
                        filemode='a',
                        level=args.log_level * 10,  # 设置日志级别
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')   # 设置日志格式
    

    context.set_context(
        mode=context.GRAPH_MODE if args.run_mode == "GRAPH_MODE" else context.PYNATIVE_MODE,
        device_target=args.device_type,
        device_id=args.device_id,
        jit_config={"jit_level": "O2"}
    )

    logging.info("run_mode:{}\ndevice_type:{}\ndevice_id:{}\nckpt_path:{}\ncamera:{}\nspeed_first:{}\ninput_path:{}\noutput_path:{}\n".format(\
        args.run_mode, args.device_type, args.device_id, args.ckpt_path, args.camera, args.speed_first, args.input_path, args.output_path))
    
    # 模型权重下载
    url = "https://modelers.cn/coderepo/web/v1/file/zhaoyu/DctNet/main/media/dct-net.ckpt"
    path = args.ckpt_path
    download(url=url, path=path, replace=True)
    print(f"模型权重下载成功")

    # 加载模型
    network = Generator(img_channels=3)
    mindspore.load_checkpoint(args.ckpt_path, network)
    network.set_train(mode=False)
    if args.camera == "yes":
        logging.info('*' * 50 + "use camera" + '*' * 50)
        camera(network, args, watermark=True)
        exit(0)
    if args.input_path.endswith(".mp4"):
        logging.info('*' * 50 + "proc local video" + '*' * 50)
        proc_video(network, args)
        exit(0)
    if args.input_path.endswith(".png") or args.input_path.endswith(".jpg"):
        logging.info('*' * 50 + "proc local image" + '*' * 50)
        proc_img(network, args)
        exit(0)
