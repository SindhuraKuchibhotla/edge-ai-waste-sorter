# SPDX-License-Identifier: MPL-2.0

"""Hardware-independent lane decisions for stock object detections."""

from collections import Counter, deque
from dataclasses import dataclass
from math import isfinite
from threading import Lock


COMPOST_LABELS = frozenset({"cup"})
RECYCLE_LABELS = frozenset({"bottle"})


def lane_for(label: str) -> str:
    label = label.strip().casefold()
    if label in COMPOST_LABELS:
        return "compost"
    if label in RECYCLE_LABELS:
        return "recycle"
    return "trash"


@dataclass(frozen=True)
class Decision:
    label: str | None
    lane: str
    model_confidence: float | None

    def __str__(self) -> str:
        if self.label is None:
            return "object=none lane=clear model_confidence=none"
        return (
            f"object={self.label} lane={self.lane} "
            f"model_confidence={self.model_confidence:.2f}"
        )


def best_detection(detections: dict) -> tuple[str, float] | None:
    """Pick the strongest result from the brick's label -> detections shape."""
    if not isinstance(detections, dict):
        return None

    best = None
    for label, values in detections.items():
        if not isinstance(label, str) or not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, dict):
                continue
            confidence = value.get("confidence")
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not isfinite(confidence)
                or not 0 <= confidence <= 1
            ):
                continue
            if best is None or confidence > best[1]:
                normalized_label = label.strip().casefold()
                if normalized_label:
                    best = (normalized_label, float(confidence))
    return best


class DetectionVoter:
    def __init__(
        self, window_s: float = 1.0, min_votes: int = 3, clear_s: float = 1.5
    ) -> None:
        self.window_s = window_s
        self.min_votes = min_votes
        self.clear_s = clear_s
        self._samples = deque()
        self._last_seen = None
        self._printed_label = None
        self._lock = Lock()

    def ingest(self, detections: dict, now: float) -> Decision | None:
        selected = best_detection(detections)
        if selected is None:
            return None

        label, confidence = selected
        with self._lock:
            self._last_seen = now
            self._samples.append((now, label, confidence))
            self._prune(now)

            counts = Counter(sample[1] for sample in self._samples)
            winner, votes = counts.most_common(1)[0]
            if votes < self.min_votes or winner == self._printed_label:
                return None

            self._printed_label = winner
            scores = [score for _, name, score in self._samples if name == winner]
            return Decision(winner, lane_for(winner), sum(scores) / len(scores))

    def expire(self, now: float) -> Decision | None:
        with self._lock:
            if self._last_seen is None or now - self._last_seen < self.clear_s:
                return None
            self._samples.clear()
            self._last_seen = None
            if self._printed_label is None:
                return None
            self._printed_label = None
            return Decision(None, "clear", None)

    def _prune(self, now: float) -> None:
        while self._samples and now - self._samples[0][0] > self.window_s:
            self._samples.popleft()
