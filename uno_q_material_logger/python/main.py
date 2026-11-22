"""Classify USB-camera frames on the UNO Q and save one result each second."""

from pathlib import Path

from arduino.app_peripherals.camera import Camera
from arduino.app_utils import App, Bridge

from display import DisplayPublisher
from material_logger import Config, MaterialClassifier, MaterialLogger, ResultWriter


APP_ROOT = Path(__file__).resolve().parent.parent
config = Config.from_file(APP_ROOT / "config.json")
logger = MaterialLogger(
    config=config,
    camera_factory=lambda: Camera(resolution=config.camera_resolution),
    classifier_factory=lambda: MaterialClassifier(
        APP_ROOT / "model" / "trashnet-mobilenetv3-small.onnx",
        APP_ROOT / "model" / "labels.json",
        config,
    ),
    writer=ResultWriter(APP_ROOT / "data"),
    publisher=DisplayPublisher(Bridge.notify),
)

App.run(user_loop=logger.loop)
