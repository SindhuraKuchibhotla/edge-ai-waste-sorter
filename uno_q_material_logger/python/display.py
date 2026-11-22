"""Translate material logger records into the OLED sketch's Bridge calls."""

from math import isfinite


class DisplayPublisher:
    def __init__(self, notify):
        self.notify = notify
        self.last_status = None
        self.item_open = False   # an item is on the belt (empty-belt check on)

    def publish(self, result: dict) -> None:
        material = result.get("material")
        lane = result.get("lane")
        confidence = result.get("confidence")
        if (
            result.get("camera_status") == "ok"
            and result.get("error") is None
            and isinstance(material, str)
            and material
            and lane in {"compost", "recycle", "trash"}
            and isinstance(confidence, (int, float))
            and not isinstance(confidence, bool)
            and isfinite(confidence)
        ):
            present = result.get("item_present")
            if present is False:          # empty belt: send nothing, the OLED goes idle
                self.item_open = False
                self.last_status = None
                return
            if present is True and not self.item_open:
                self.notify("display_scanning")   # a new item arrived
            self.item_open = present is True
            percent = max(0, min(100, round(confidence * 100)))
            inference_ms = result.get("inference_ms")
            self.notify("show_result", lane.upper(), material, percent, inference_ms)
            self.last_status = None
            return

        status = "MODEL ERROR" if str(result.get("error") or "").startswith("model_error:") else "NO CAMERA"
        if status != self.last_status:
            self.notify("show_status", status)
            self.last_status = status
