"""Portable inference and persistent logging for the UNO Q material app."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from math import isfinite
from pathlib import Path
from time import monotonic, sleep

import numpy as np
import onnxruntime as ort
from PIL import Image


RECYCLE_MATERIALS = frozenset({"cardboard", "glass", "metal", "paper"})


def lane_for(material: str, confidence: float) -> str:
    """Turn the top model prediction into the MVP disposal lane."""
    if not isfinite(confidence):
        raise ValueError("model confidence must be finite")
    if material == "food_organic":
        return "compost"
    if material in RECYCLE_MATERIALS:
        return "recycle"
    return "trash"


@dataclass(frozen=True)
class Config:
    camera_resolution: tuple[int, int]
    roi_fraction: float
    roi_center_x: float
    roi_center_y: float
    detect_empty_belt: bool = True     # only publish while something is on the belt
    presence_threshold: float = 12.0   # BeltGate difference that counts as an item

    @classmethod
    def from_file(cls, path: Path) -> "Config":
        values = json.loads(path.read_text())
        resolution = tuple(values["camera_resolution"])
        if len(resolution) != 2 or any(not isinstance(size, int) or size <= 0 for size in resolution):
            raise ValueError("camera_resolution must contain two positive integers")
        fraction = float(values["roi_fraction"])
        center_x = float(values["roi_center_x"])
        center_y = float(values["roi_center_y"])
        if not 0 < fraction <= 1 or not 0 <= center_x <= 1 or not 0 <= center_y <= 1:
            raise ValueError("ROI fraction must be in (0, 1], centers in [0, 1]")
        detect = bool(values.get("detect_empty_belt", True))
        threshold = float(values.get("presence_threshold", 12.0))
        if threshold <= 0:
            raise ValueError("presence_threshold must be positive")
        return cls(resolution, fraction, center_x, center_y, detect, threshold)


class BeltGate:
    """Is something on the belt? Compares the crop with a learned empty-belt picture.

    The first `learn_frames` frames after start are taken as the empty belt, so start
    the app with the belt empty. While the belt is empty the picture keeps adapting
    slowly to light changes. Each result logs `difference`: note it with the belt empty
    and with an item, and set `presence_threshold` in config.json between the two.
    """

    def __init__(self, threshold: float = 12.0, learn_frames: int = 3, adapt: float = 0.05):
        self.threshold = threshold
        self.learn_frames = learn_frames
        self.adapt = adapt
        self.background = None
        self.seen = 0
        self.last_difference = 0.0

    def update(self, crop_bgr: np.ndarray) -> bool:
        gray = Image.fromarray(np.ascontiguousarray(crop_bgr[:, :, ::-1])).convert("L")
        small = np.asarray(gray.resize((48, 48), Image.Resampling.BILINEAR), dtype=np.float32)
        self.seen += 1
        if self.background is None:
            self.background = small
            return False
        self.last_difference = float(np.mean(np.abs(small - self.background)))
        if self.seen <= self.learn_frames:
            self.background = (self.background * (self.seen - 1) + small) / self.seen
            return False
        present = self.last_difference > self.threshold
        if not present:
            self.background = (1 - self.adapt) * self.background + self.adapt * small
        return present


def crop_region(frame: np.ndarray, config: Config) -> np.ndarray:
    height, width = frame.shape[:2]
    side = max(1, round(min(height, width) * config.roi_fraction))
    left = max(0, min(width - side, round(width * config.roi_center_x - side / 2)))
    top = max(0, min(height - side, round(height * config.roi_center_y - side / 2)))
    return frame[top:top + side, left:left + side]


def prepare_input(frame: np.ndarray, config: Config) -> np.ndarray:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("camera frame must be an H×W×3 BGR array")
    crop = crop_region(frame, config)
    image = Image.fromarray(crop[:, :, ::-1]).resize((256, 256), Image.Resampling.BILINEAR)
    image = image.crop((16, 16, 240, 240))
    pixels = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    pixels = (pixels - np.array((0.485, 0.456, 0.406), dtype=np.float32)) / np.array(
        (0.229, 0.224, 0.225), dtype=np.float32
    )
    return np.transpose(pixels, (2, 0, 1))[None].copy()


class MaterialClassifier:
    def __init__(self, model_path: Path, labels_path: Path, config: Config):
        metadata = json.loads(labels_path.read_text())
        self.labels = metadata["labels"]
        self.config = config
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        output_size = self.session.get_outputs()[0].shape[-1]
        if output_size != len(self.labels):
            raise ValueError("model output does not match label count")
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, frame: np.ndarray) -> tuple[str, float]:
        logits = self.session.run(None, {self.input_name: prepare_input(frame, self.config)})[0][0]
        shifted = logits - np.max(logits)
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        best = int(np.argmax(probabilities))
        return self.labels[best], float(probabilities[best])


class ResultWriter:
    def __init__(self, data_dir: Path, *, started_at: datetime | None = None):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        started_at = started_at or datetime.now(timezone.utc)
        stamp = started_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%fZ")
        for sequence in range(1000):
            suffix = f"-{sequence}" if sequence else ""
            path = self.data_dir / f"results-{stamp}{suffix}.txt"
            try:
                path.touch(exist_ok=False)
            except FileExistsError:
                continue
            self.path = path
            break
        else:
            raise RuntimeError("could not reserve a unique results file")

    def append(self, result: dict) -> Path:
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(result, separators=(",", ":")) + "\n")
        return self.path


class MaterialLogger:
    def __init__(self, *, config: Config, camera_factory, classifier_factory,
                 writer: ResultWriter, publisher=None, retry_seconds: float = 5.0):
        self.config = config
        self.camera_factory = camera_factory
        self.classifier_factory = classifier_factory
        self.writer = writer
        self.publisher = publisher
        self.gate = None
        self.retry_seconds = retry_seconds
        self.started_at = monotonic()
        self.camera = None
        self.classifier = None
        self.next_camera_retry = 0.0
        self.camera_error = "camera has not opened yet"

    def _close_camera(self) -> None:
        camera, self.camera = self.camera, None
        if camera is not None:
            try:
                camera.stop()
            except Exception:
                pass

    def _open_camera(self, now: float) -> None:
        if now < self.next_camera_retry:
            return
        camera = None
        try:
            camera = self.camera_factory()
            camera.start()
            self.camera = camera
            self.camera_error = ""
        except Exception as exc:
            if camera is not None:
                try:
                    camera.stop()
                except Exception:
                    pass
            self.camera_error = f"{type(exc).__name__}: {exc}"
            self.next_camera_retry = now + self.retry_seconds

    def sample(self, *, now: float | None = None, timestamp: datetime | None = None) -> dict:
        now = monotonic() if now is None else now
        timestamp = datetime.now(timezone.utc) if timestamp is None else timestamp
        result = {
            "timestamp_utc": timestamp.isoformat().replace("+00:00", "Z"),
            "uptime_s": round(max(0.0, now - self.started_at), 3),
            "camera_status": "unavailable",
            "material": None,
            "confidence": None,
            "lane": None,
            "inference_ms": None,
            "error": None,
        }
        if self.camera is None:
            self._open_camera(now)
        if self.camera is None:
            result["error"] = self.camera_error
            return result
        try:
            frame = self.camera.capture()
            if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                raise ValueError("camera returned no frame")
            result["camera_status"] = "ok"
            if self.config.detect_empty_belt:
                if self.gate is None:
                    self.gate = BeltGate(self.config.presence_threshold)
                result["item_present"] = self.gate.update(crop_region(frame, self.config))
                result["difference"] = round(self.gate.last_difference, 2)
        except Exception as exc:
            result["camera_status"] = "read_error"
            result["error"] = f"{type(exc).__name__}: {exc}"
            self._close_camera()
            self.next_camera_retry = now + self.retry_seconds
            self.camera_error = result["error"]
            return result
        try:
            if self.classifier is None:
                self.classifier = self.classifier_factory()
            inference_started = monotonic()
            material, confidence = self.classifier.predict(frame)
            lane = lane_for(material, confidence)
            result["material"] = material
            result["confidence"] = round(confidence, 6)
            result["lane"] = lane
            result["inference_ms"] = max(0, round((monotonic() - inference_started) * 1000))
        except Exception as exc:
            result["error"] = f"model_error: {type(exc).__name__}: {exc}"
        return result

    def loop(self) -> None:
        started = monotonic()
        result = self.sample(now=started)
        self.writer.append(result)
        print(json.dumps(result), flush=True)
        if self.publisher is not None:
            try:
                self.publisher.publish(result)
            except Exception as exc:
                print(f"display_publish_error={type(exc).__name__}: {exc}", flush=True)
        sleep(max(0.0, 1.0 - (monotonic() - started)))
