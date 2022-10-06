#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import json
import os
from os import write
import subprocess as sp
import time
from loguru import logger

import cv2

import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
# 模型的名字和路径
model_name = "yolox-s"
model_path = "model/yolox_s.pth"
# 使用GPU来进行预测
device = "gpu"
# 是否开启fp16
fp16 = False
# 检测的置信度
conf = 0.3
nmsthre = 0.45
# 测试图片的大小
test_size = (640, 640)
# 原始rtmp流和检测后的rtmp流
origin_rtmp = "rtmp://192.168.1.30:8100/live/origin"
new_rtmp = "rtmp://192.168.1.30:8100/live/detect"

# 代码预测
class Predictor(object):
    def __init__(
            self,
            model,
            cls_names=COCO_CLASSES,
            device="cpu",
            fp16=False,
            legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = None
        self.num_classes = 80
        self.confthre = conf
        self.nmsthre = nmsthre
        self.test_size = test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)

    # 检测接口
    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None
        # 获取图片的一些信息
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio
        # 图片预处理
        img, _ = self.preproc(img, None, self.test_size)
        # 把图片转换为tensor
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16
        # 调用模型获取结果
        with torch.no_grad():
            # t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            # logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info
    # 结果可视化
    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()
        # 获取边框、置信度、标签等信息
        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res

# ffmpeg视频转换
class ffmpeg_writer:
    def __init__(self, rtmpUrl, fps, width, height):
        # ffmpeg command
        command = ['ffmpeg',
                   '-y',
                   '-f', 'rawvideo',
                   '-vcodec', 'rawvideo',
                   '-pix_fmt', 'bgr24',
                   '-s', "{}x{}".format(width, height),
                   '-r', str(fps),
                   '-i', '-',
                   '-c:v', 'libx264',
                   '-pix_fmt', 'yuv420p',
                   # '-preset', 'ultrafast',
                   '-f', 'flv',
                   # '-flvflags', 'no_duration_filesize',
                   rtmpUrl]
        # 启动一个管道
        self.pipe = sp.Popen(command, stdin=sp.PIPE)

    def write(self, frame):
        # 往管道中写入数据
        self.pipe.stdin.write(frame.tostring())

    def release(self):
        # 管道释放
        self.pipe.terminate()


# outputs, img_info = predictor.inference("test.jpg")
# result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
# cv2.imwrite("res.jpg", result_image)


def imageflow_demo(predictor, callback):
    # 获取串流视频的帧率和视频的宽高信息
    cap = cv2.VideoCapture(origin_rtmp)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 初始化我们自定义的管道
    writer = ffmpeg_writer(new_rtmp, fps, width, height)
    # 不断循环获取到当前帧的信息
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            # 使用模型进行预测
            outputs, img_info = predictor.inference(frame)
            # 结果可视化
            if outputs[0] is not None:
                # 把模型输出的结果转换为numpy格式
                info = outputs[0].cpu().numpy()
                # 获取各项信息
                bboxes = info[:, 0:4]
                cls = info[:, 6]
                scores = info[:, 4] * info[:, 5]
                # 组装最后的结果
                detect = []
                for i in range(0, len(bboxes)):
                    detect.append({
                        "box": bboxes[i].tolist(),
                        "label": COCO_CLASSES[int(cls[i].astype(int))],
                        "score": scores[i].astype(float)
                    })
                # 对结果进行回调
                callback(json.dumps(detect))
            # 绘制边框信息
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            # 把检测后的结果写入到管道中
            writer.write(result_frame)
    # writer.release()


# 视频转码处理，需要另外新开一个线程运行
def video_process(callback):
    # 先设置一些必要的参数
    exp = get_exp(None,  model_name)
    exp.test_conf = conf
    exp.nmsthre = nmsthre
    exp.test_size = test_size
    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.cuda()
    if fp16:
        model.half()  # to FP16
    model.eval()
    logger.info("loading checkpoint")
    # 加载我们的模型
    ckpt = torch.load(model_path, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")
    # 初始化预测器
    predictor = Predictor(model, COCO_CLASSES, device, fp16, False)
    # 视频实时处理
    imageflow_demo(predictor, callback)
