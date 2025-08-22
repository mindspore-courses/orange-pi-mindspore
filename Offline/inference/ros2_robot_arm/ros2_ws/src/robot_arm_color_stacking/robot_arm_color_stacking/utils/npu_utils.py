import cv2
import numpy as np
import torch

from .det_utils import letterbox, scale_coords, nms


def preprocess_image(img_bgr, cfg):
    img, scale_ratio, pad_size = letterbox(img_bgr, new_shape=cfg["input_shape"])
    # bgr2rgb, HWC2CHW
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float16) / 255.0
    return img, scale_ratio, pad_size


def draw_bbox(bbox, img0, color, wt, names):
    det_result_str = ""
    for idx, class_id in enumerate(bbox[:, 5]):
        if float(bbox[idx][4] < float(0.05)):
            continue
        img0 = cv2.rectangle(
            img0,
            (int(bbox[idx][0]), int(bbox[idx][1])),
            (int(bbox[idx][2]), int(bbox[idx][3])),
            color,
            wt,
        )
        img0 = cv2.putText(
            img0,
            str(idx) + " " + names[int(class_id)],
            (int(bbox[idx][0]), int(bbox[idx][1] + 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
        img0 = cv2.putText(
            img0,
            "{:.4f}".format(bbox[idx][4]),
            (int(bbox[idx][0]), int(bbox[idx][1] + 32)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
        det_result_str += "{} {} {} {} {} {}\n".format(
            names[bbox[idx][5]],
            str(bbox[idx][4]),
            bbox[idx][0],
            bbox[idx][1],
            bbox[idx][2],
            bbox[idx][3],
        )
    return img0


def get_labels_from_txt(path):
    labels_dict = dict()
    with open(path) as f:
        for cat_id, label in enumerate(f.readlines()):
            labels_dict[cat_id] = label.strip()
    return labels_dict


def draw_prediction(pred, img_bgr, labels):
    img_dw = draw_bbox(pred, img_bgr, (0, 255, 0), 2, labels)
    return img_dw


def infer_image(img_bgr, model, class_names, cfg):
    img, scale_ratio, pad_size = preprocess_image(img_bgr, cfg)
    output = model.infer([img])[0]

    output = torch.tensor(output)
    boxout = nms(output, conf_thres=cfg["conf_thres"], iou_thres=cfg["iou_thres"])
    pred_all = boxout[0].numpy()
    scale_coords(
        cfg["input_shape"],
        pred_all[:, :4],
        img_bgr.shape,
        ratio_pad=(scale_ratio, pad_size),
    )
    print("pred_all partial shape:", pred_all[:, :4].shape)
    print(img_bgr.shape)
    drawed_res = draw_prediction(pred_all, img_bgr, class_names)
    return pred_all, class_names, drawed_res


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    # x center
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2
    # y center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2
    # width
    y[..., 2] = x[..., 2] - x[..., 0]
    # height
    y[..., 3] = x[..., 3] - x[..., 1]
    return y
