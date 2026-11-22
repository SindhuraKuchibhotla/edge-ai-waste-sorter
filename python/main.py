# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, UTC
from threading import Thread
from time import monotonic, sleep

from arduino.app_utils import App
from arduino.app_bricks.web_ui import WebUI
from arduino.app_bricks.video_objectdetection import VideoObjectDetection

from sorting import DetectionVoter

ui = WebUI()
detection_stream = VideoObjectDetection(confidence=0.5, debounce_sec=0.0)
voter = DetectionVoter()
first_callback = True

ui.on_message("override_th", lambda sid, threshold: detection_stream.override_threshold(threshold))

def on_detections(detections: dict) -> None:
    global first_callback

    if first_callback:
        print(f"first_detection_callback={detections!r}", flush=True)
        first_callback = False

    # Keep the example's browser history and live video behavior.
    for label, values in detections.items():
        for value in values:
            entry = {
                "content": label,
                "confidence": value.get("confidence"),
                "timestamp": datetime.now(UTC).isoformat(),
            }
            ui.send_message("detection", message=entry)

    decision = voter.ingest(detections, monotonic())
    if decision is not None:
        print(decision, flush=True)


def clear_when_camera_quiet() -> None:
    # The brick invokes callbacks for detections, so a timer handles removal.
    while True:
        sleep(0.1)
        decision = voter.expire(monotonic())
        if decision is not None:
            print(decision, flush=True)

detection_stream.on_detect_all(on_detections)
Thread(target=clear_when_camera_quiet, daemon=True).start()

App.run()
