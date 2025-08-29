"""
Copyright 2022 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import time

import cv2
import numpy as np
import torch
import torchvision


def letterbox(
    img,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=False,
    scaleFill=False,
    scaleup=True,
):
    # Resize image to a 32-pixel-multiple rectangle
    # current shape [height, width]
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # only scale down, do not scale up (for better test mAP)
    if not scaleup:
        r = min(r, 1.0)

    # Compute padding
    # width, height ratios
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    # wh padding
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    # minimum rectangle
    if auto:
        # wh padding
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)
    # stretch
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        # width, height ratios
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    # divide padding into 2 sides
    dw /= 2
    dh /= 2

    # resize
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # add border
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return img, ratio, (dw, dh)


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    # x center
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    # y center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    # width
    y[:, 2] = x[:, 2] - x[:, 0]
    # height
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    # number of masks
    nm=0,
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # YOLOv5 model in validation model, output = (inference_out, loss_out)
    if isinstance(prediction, (list, tuple)):
        # select only inference output
        prediction = prediction[0]

    device = prediction.device
    # Apple MPS
    mps = "mps" in device.type
    # MPS not fully supported yet, convert tensors to CPU before NMS
    if mps:
        prediction = prediction.cpu()
    # batch size
    bs = prediction.shape[0]
    # number of classes
    nc = prediction.shape[2] - nm - 5
    # candidates
    xc = prediction[..., 4] > conf_thres

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    # (pixels) maximum box width and height
    max_wh = 7680
    # maximum number of boxes into torchvision.ops.nms()
    max_nms = 30000
    # seconds to quit after
    time_limit = 0.5 + 0.05 * bs
    # require redundant detections
    redundant = True
    # multiple labels per box (adds 0.5ms/img)
    multi_label &= nc > 1
    # use merge-NMS
    merge = False

    t = time.time()
    # mask start index
    mi = 5 + nc
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    # image index, image inference
    for xi, x in enumerate(prediction):
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        # confidence
        x = x[xc[xi]]

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            # box
            v[:, :4] = lb[:, 1:5]
            # conf
            v[:, 4] = 1.0
            # cls
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        # conf = obj_conf * cls_conf
        x[:, 5:] *= x[:, 4:5]

        # Box/Mask
        # center_x, center_y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        # zero columns if no masks
        mask = x[:, mi:]

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        # best class only
        else:
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        # number of boxes
        n = x.shape[0]
        # no boxes
        if not n:
            continue
        # excess boxes
        elif n > max_nms:
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        else:
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        # classes
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        # NMS
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        # limit detections
        if i.shape[0] > max_det:
            i = i[:max_det]
        # Merge NMS (boxes merged using weighted mean)
        if merge and (1 < n < 3e3):
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            # iou matrix
            iou = box_iou(boxes[i], boxes) > iou_thres
            # box weights
            weights = iou * scores[None]
            # merged boxes
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                1, keepdim=True
            )
            if redundant:
                # require redundancy
                i = i[iou.sum(1) > 1]

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            # time limit exceeded
            break

    return output


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    # top left x
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    # top left y
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    # bottom right x
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    # bottom right y
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    # calculate from img0_shape
    if ratio_pad is None:
        # gain  = old / new
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        # wh padding
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain
        ) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    # x padding
    coords[:, [0, 2]] -= pad[0]
    # y padding
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    # faster individually
    if isinstance(boxes, torch.Tensor):
        # x1
        boxes[:, 0].clamp_(0, shape[1])
        # y1
        boxes[:, 1].clamp_(0, shape[0])
        # x2
        boxes[:, 2].clamp_(0, shape[1])
        # y2
        boxes[:, 3].clamp_(0, shape[0])
    # np.array (faster grouped)
    else:
        # x1, x2
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])
        # y1, y2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])


def nms(box_out, conf_thres=0.4, iou_thres=0.5):
    try:
        boxout = non_max_suppression(
            box_out, conf_thres=conf_thres, iou_thres=iou_thres, multi_label=True
        )
    except:
        boxout = non_max_suppression(
            box_out, conf_thres=conf_thres, iou_thres=iou_thres
        )
    return boxout
