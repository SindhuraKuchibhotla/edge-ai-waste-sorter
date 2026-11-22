# Garbage Collector â€” Stage A and local camera testing

This Arduino UNO Q App Lab prototype uses the stock USB-camera object detector.
It retains live video in the browser and prints a stable object label, waste
lane, and model confidence in App Lab's **Main (Python)** console. It assumes
one item is in view at a time. It does not control a diverter or LEDs yet.

The camera page and brick setup come from Arduino's [Detect Objects on Camera
example](https://github.com/arduino/app-bricks-examples/tree/main/inspirational/common/video-generic-object-detection)
at commit `c3dfa96249cd49239cccf36b1cdbcf19023c5a35`. The copied Arduino
files retain their MPL-2.0 notices; `LICENSE.txt` and the bundled font license
files are included.

## Test with a camera on your Mac

You can develop and check the detector without the UNO Q connected. The local
runner uses Megvii's pretrained **YOLOX Nano COCO ONNX** model, draws live
bounding boxes and labels in an OpenCV window, and prints stable waste-lane
changes in Terminal. It reuses `python/sorting.py`, so the current rule is
**cup â†’ compost, bottle â†’ recycle, every other detected label â†’ trash**. This
is a general object detector; it has not been trained on our waste items.

Install Python 3.12, then from the repo root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r local_test/requirements.txt
mkdir -p local_test/models
curl -fL --retry 3 -o local_test/models/yolox_nano.onnx \
  https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx
python -m local_test --camera 0
```

The model download is about 3 MB and stays in the ignored `local_test/models/`
directory. Connect the USB camera before starting the runner. macOS may ask
you to allow camera access for Terminal (or your terminal app); if needed,
check **System Settings â†’ Privacy & Security â†’ Camera**. If camera 0 is not
the USB camera, try `--camera 1` or `--camera 2`. Press **q** or **Esc** in the
preview window to stop. A box shows the raw YOLOX label, confidence, and
mapped lane. The banner and Terminal output show the lane after voting.

To check inference before the camera is available, use a still image:

```sh
python -m local_test --image /path/to/photo.jpg
```

This prints each detected object and saves an annotated image to
`local_test/output/annotated.jpg`. A single image cannot produce a stable
three-vote decision. Use `--confidence 0.4`, `--min-votes 3`, `--window-s 1`,
or `--clear-s 1.5` to tune live testing. For example, lower confidence if a
cup is missed, then check whether false detections increase.

The local ONNX weights and the UNO Q brick's Edge Impulse `.eim` file are
separate deployments of the YOLOX Nano COCO model family. Their detections
may differ. Local testing lets us develop the mapping and preview now; the
Arduino camera, brick output, and timing still need a later board test. The
local runner does not retrain or deploy a custom waste classifier.
The local detector's preprocessing and output handling are adapted from
[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), whose Apache 2.0
license is included at `local_test/LICENSE-YOLOX.txt`.

## Test the material classifier on your Mac

This is a separate experiment from the YOLOX detector and the UNO Q App Lab
brick. The default MobileNetV3 Small checkpoint predicts **cardboard,
food_organic, glass, metal, paper, plastic, or trash** for one centered item.
It was trained from an ImageNet-pretrained TrashNet baseline using food-waste,
UCI RealWaste, and TACO wrapper photos. It does not produce bounding boxes,
a disposal-lane decision, or an empty-belt result. The word
`trash` here is a dataset category, not our sorterâ€™s final disposal decision.

With Python 3.12 installed, run from the repo root:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r material_classifier/requirements.txt
python -m material_classifier.preview --camera 0
```

The seven-label checkpoint and the original six-label TrashNet baseline are
both in `material_classifier/models/`; the preview selects the seven-label
checkpoint by default. Dataset archives and training outputs stay in ignored
local folders. See [the model card](../material_classifier/MODEL_CARD.md) and
[wrapper experiment](../material_classifier/WRAPPER_EXPERIMENT.md) for current
metrics and training details. The reported
test score does not establish performance on the conveyor. If the USB
camera is not index 0, try `--camera 1`. Keep one item centered inside the
yellow preview box. Press `s` to save a full, unlabeled camera frame for later
fine-tuning, or `q` to quit. An empty view still gets a class prediction.
For a side view where the belt is off center, adjust the square region with
`--roi`, `--roi-center-x`, and `--roi-center-y` (all fractions from 0 to 1).
For example, `--roi 0.4 --roi-center-x 0.32 --roi-center-y 0.2` focuses on
an item toward the upper-left of a 1920Ã—1080 side view. Check the yellow
preview box after moving the camera; a crop that misses part of the item can
change the prediction. Mounting the camera above the belt so it fills the
frame will give a more representative test.
For a camera check without an OpenCV window, use
`python -m material_classifier.preview --camera 0 --once`; it saves one raw
frame under `material_classifier/captures/` and prints the seven scores.

To check the checkpoint on one image without a camera:

```sh
python -m material_classifier.preview --image material_classifier/data/dataset-resized/glass/glass1.jpg
```

For a fast training-pipeline check, run `python -m material_classifier.download`
and then `python -m material_classifier.train --max-per-class 12 --epochs 1`.
This is only a smoke test. Once we have labeled belt images organized into
class-named folders, the same script can fine-tune from this checkpoint:

```sh
python -m material_classifier.train \
  --data-dir material_classifier/data/belt-labeled \
  --start-from material_classifier/models/food-organic-mobilenetv3-small.pt \
  --unfreeze-last 2 --learning-rate 0.0001 \
  --output-dir material_classifier/output/belt
```

If belt images introduce new labels, training preserves existing output rows
and initializes rows for the new labels. For a credible
belt evaluation, prepare `belt-labeled/train/<class>/`,
`belt-labeled/validation/<class>/`, and `belt-labeled/test/<class>/` folders
instead of relying on the script's random split. Put every photo of one
physical item into only one split; repeated frames of the same item across
splits would inflate the result. Each split needs at least one image of each
class. The script also accepts flat class folders for quick experiments.
After collecting frames with `s`, review and copy them into a selected split
with `python -m material_classifier.label_captures --split train`. Press the
number shown for the correct label, `n` to skip, or `q` to stop. Repeat with
`--split validation` or `--split test` for different physical items. The
capture files remain untouched; already labeled filenames are skipped so the
same frame cannot be copied into two splits. The default label menu includes
the seven current classes; use `--classes` to choose a different set, and
keep that set consistent across all splits. The seven-class ONNX export is
included in the UNO Q material logger package; it has not been deployed to
the board in this training run. See [the model card](../material_classifier/MODEL_CARD.md)
for metrics and limitations.

## Package the trained material model and OLED display for UNO Q

The separate [Material Classifier OLED](../uno_q_material_logger/README.md)
app contains the same trained checkpoint exported to ONNX. It starts at boot
once **Run at startup** is enabled in App Lab, reads a USB camera without a
laptop connected, writes camera health, material label, and model score to a
new text file for each start, and sends successful predictions to the OLED
sketch through RouterBridge. Build its App Lab import archive with:

```sh
python -m material_classifier.package_uno_q
```

This writes `dist/material-classifier-oled-uno-q.zip`. The package includes the
OLED sketch from `sketch/`. It does not replace this repository's stock
object-detection app.

## Local checks

From this repo, run:

```sh
python3 -m unittest discover -s tests -v
python3 -m py_compile python/main.py python/sorting.py
```

These checks do not require the board, camera, or Arduino Python packages.

## Test on the UNO Q

The Mac runner and UNO Q app are separate. For the board test, move the webcam
from the Mac to a powered USB-C hub connected to the UNO Q. Keep the hub powered
and plug in the webcam **before** starting the app. The Mac runs App Lab and
shows the browser page; the UNO Q performs camera inference.

Create an App Lab import archive from the repo root. The `app.yaml` file must
be at the root of the ZIP. Rebuild it after changing board app files:

```sh
mkdir -p dist
zip -q -r dist/waste-sorter-uno-q.zip app.yaml README.md LICENSE.txt \
  python/main.py python/sorting.py sketch/sketch.ino sketch/sketch.yaml assets
```

1. In Arduino App Lab on the Mac, connect to the UNO Q over the local network.
   Complete the initial board setup first if App Lab asks for a USB-C data
   connection. Keep the Mac and UNO Q on the same network.
2. In **My Apps**, use **Import App** (it may be under **Create New App**)
   and select `dist/waste-sorter-uno-q.zip`. Select the imported **Garbage
   Collector** app and press **Run**. The first run can take longer while App
   Lab prepares its bricks.
3. Open the live page if it does not launch automatically. It is normally at
   `http://<board-name>.local:7000`. Look for the camera stream and object
   boxes/labels.
4. Open App Lab's **Main (Python)** console. Its first detection callback is
   printed once as `first_detection_callback=...` for checking the installed
   brick's result format.
5. Show a cup, a bottle, and an unmapped object one at a time. Expect
   `compost`, `recycle`, and `trash`. Remove the item and expect one
   `object=none lane=clear` line after about 1.5 seconds.

The output format is `object=cup lane=compost model_confidence=0.86`.
`model_confidence` is the detector's mean score for the winning label, **not**
the probability that the waste lane is correct. Cup maps to compost, bottle
maps to recycle, and every other stock label falls back to trash. The initial
tuning values are a 1-second vote window, at least 3 matching detections, and
a 1.5-second clear timeout. Adjust them in
`python/sorting.py` after the live test if needed.
