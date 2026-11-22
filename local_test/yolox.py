# Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
# Modified for this project's local camera runner. See LICENSE-YOLOX.txt.

"""Small ONNX Runtime adapter for Megvii's YOLOX Nano COCO model.

Preprocessing, output decoding, and NMS follow Megvii's Apache-2.0 ONNX demo:
https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/ONNXRuntime
"""

from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort


# COCO class order from Megvii's yolox/data/datasets/coco_classes.py.
COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]


def _prepare(frame: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, float]:
    height, width = frame.shape[:2]
    ratio = min(shape[0] / height, shape[1] / width)
    resized = cv2.resize(frame, (int(width * ratio), int(height * ratio)))
    padded = np.full((shape[0], shape[1], 3), 114, dtype=np.uint8)
    padded[: resized.shape[0], : resized.shape[1]] = resized
    tensor = np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)
    return tensor[None], ratio


def _decode(output: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    grids = []
    strides = []
    for stride in (8, 16, 32):
        y, x = np.meshgrid(
            np.arange(shape[0] // stride), np.arange(shape[1] // stride), indexing="ij"
        )
        grids.append(np.stack((x, y), axis=-1).reshape(-1, 2))
        strides.append(np.full((grids[-1].shape[0], 1), stride))
    grid = np.concatenate(grids)
    stride = np.concatenate(strides)
    predictions = output[0].copy()
    if predictions.shape != (len(grid), 5 + len(COCO_CLASSES)):
        raise ValueError(
            f"Expected raw YOLOX COCO output ({len(grid)}, 85), "
            f"got {predictions.shape}. Check the model file."
        )
    predictions[:, :2] = (predictions[:, :2] + grid) * stride
    predictions[:, 2:4] = np.exp(predictions[:, 2:4]) * stride
    return predictions


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        index = int(order[0])
        keep.append(index)
        if order.size == 1:
            break
        rest = order[1:]
        upper_left = np.maximum(boxes[index, :2], boxes[rest, :2])
        lower_right = np.minimum(boxes[index, 2:], boxes[rest, 2:])
        size = np.maximum(lower_right - upper_left, 0)
        intersection = size[:, 0] * size[:, 1]
        area_one = np.prod(np.maximum(boxes[index, 2:] - boxes[index, :2], 0))
        area_rest = np.prod(np.maximum(boxes[rest, 2:] - boxes[rest, :2], 0), axis=1)
        union = area_one + area_rest - intersection
        overlap = np.divide(
            intersection, union, out=np.zeros_like(intersection), where=union > 0
        )
        order = rest[overlap <= threshold]
    return keep


class YoloXNano:
    def __init__(self, model_path: str, confidence: float = 0.5) -> None:
        self.confidence = confidence
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        input_info = self.session.get_inputs()[0]
        shape = input_info.shape
        if len(shape) != 4 or shape[0] != 1 or shape[1] != 3 or not all(
            isinstance(value, int) and value > 0 for value in shape[2:]
        ):
            raise ValueError(f"Expected a static NCHW YOLOX model, got input {shape}")
        self.shape = (shape[2], shape[3])
        self.input_name = input_info.name

    def detect(self, frame: np.ndarray) -> list[Detection]:
        tensor, ratio = _prepare(frame, self.shape)
        output = self.session.run(None, {self.input_name: tensor})[0]
        predictions = _decode(output, self.shape)
        scores = predictions[:, 4, None] * predictions[:, 5:]
        class_indices = scores.argmax(axis=1)
        confidences = scores[np.arange(len(scores)), class_indices]
        selected = confidences >= self.confidence
        if not np.any(selected):
            return []

        centers = predictions[selected, :2]
        sizes = predictions[selected, 2:4]
        boxes = np.concatenate((centers - sizes / 2, centers + sizes / 2), axis=1)
        boxes /= ratio
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, frame.shape[1] - 1)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, frame.shape[0] - 1)
        selected_scores = confidences[selected]
        selected_classes = class_indices[selected]
        return [
            Detection(COCO_CLASSES[int(selected_classes[i])], float(selected_scores[i]),
                      tuple(int(value) for value in boxes[i]))
            for i in _nms(boxes, selected_scores, threshold=0.45)
        ]
