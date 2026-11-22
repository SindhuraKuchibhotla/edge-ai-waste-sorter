# Material classifier model card

This is a one-item MobileNetV3 Small image classifier. The current checkpoint
predicts `cardboard`, `food_organic`, `glass`, `metal`, `paper`, `plastic`, or
`trash`; it does not predict disposal lanes. The original baseline data comes from
[Kaggle: feyzazkefe/trashnet](https://www.kaggle.com/datasets/feyzazkefe/trashnet),
a copy of [Gary Thung and Mindy Yang's TrashNet](https://github.com/garythung/trashnet).
The Kaggle listing reports an unknown dataset license, so the images are not
included in this repository.

## Current seven-label checkpoint

`models/food-organic-mobilenetv3-small.pt` extends the original TrashNet
checkpoint with `food_organic`. The additional photos came from the
`food_waste`, `eggshells`, and `coffee_grounds` categories of
[Alistair King's household waste dataset](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification),
and seven matching classes in [UCI RealWaste](https://archive.ics.uci.edu/dataset/908/realwaste).
Food-organic photos were deduplicated before splitting. The mixed dataset has
5,167 training, 1,107 validation, and 1,118 test images. Training preserved
the original six classifier rows, used balanced sampling, and unfroze the
last two backbone blocks for the mixed-data phase. The training record has
the commands and source split details.

The current checkpoint continues that model with 58 cropped `Crisp packet`
and `Other plastic wrapper` training images from the
[TACO dataset](https://github.com/pedropro/TACO), mapped to `trash`.
Another 20 TACO source images were reserved for evaluation. Its wrapper
training procedure, split provenance, and before/after results are in
`WRAPPER_EXPERIMENT.md`. No TACO photos are committed.

The current checkpoint scored **83.3%** overall (931/1,118) on the original
mixed held-out test set, up from 81.9% (916/1,118) before wrapper tuning.
`food_organic` recall was **184/194 (94.8%)**, and `trash` recall was
**79/96 (82.3%)**. On the 603 reserved RealWaste test photos, accuracy was
**483/603 (80.1%)**, food-organic recall **54/62 (87.1%)**, and trash recall
**64/75 (85.3%)**. The reserved TACO wrapper crops predicted `trash` in
**18/20** cases, up from 4/20. These scores measure recognition on the
listed datasets; they do not establish whether a local item is accepted by
a compost or recycling program.

The three available conveyor photos remain difficult. With the current
centered 90% crop, the model predicts the paper/cardboard cup as `paper`
(94.9%), the snack wrapper as `cardboard` (68.7%), and the plastic bottle as
`paper` (69.8%). The wrapper and bottle are still wrong. The TACO improvement
does not prove belt wrapper performance. We have no labeled food item
photographed on the belt, so belt food-organic accuracy is unknown. A
whole-frame classifier also gives a label to an empty view. Collect
fixed-camera belt photos, including food items and empty frames, before
using its outputs to actuate sorting.

## Original six-label TrashNet baseline

### Training

- Architecture: torchvision MobileNetV3 Small, initialized from its
  ImageNet-1K V1 weights. Its classifier layers were trained; the
  convolutional backbone stayed frozen.
- Input: center-cropped 224 × 224 RGB image, normalized with ImageNet values.
- Dataset: all 2,527 Kaggle images. Deterministic stratified 70/15/15 split
  with seed 42: 1,766 train, 378 validation, 383 test.
- Run: `python -m material_classifier.train` with the pinned dependencies in
  `material_classifier/requirements.txt`. Best validation checkpoint was at
  epoch 6 of 8.
- Artifacts: `models/trashnet-mobilenetv3-small.pt` is a copy of the best
  local checkpoint. `models/trashnet-mobilenetv3-small.metrics.json` records
  the held-out results and confusion matrix.

### Results on held-out TrashNet photos

Overall accuracy: **85.1%** (326 of 383 images). Per-class recall:

| Class | Recall | Test images |
| --- | ---: | ---: |
| cardboard | 95.1% | 61 |
| glass | 69.7% | 76 |
| metal | 91.9% | 62 |
| paper | 85.6% | 90 |
| plastic | 91.8% | 73 |
| trash | 66.7% | 21 |

These are white-background dataset results. They do not measure accuracy on
the conveyor. The model always chooses one of its six labels, including when
the camera sees an empty belt. There is no compost label, bounding box,
material contamination check, or disposal-policy mapping. Probabilities are
uncalibrated model scores, not verified sorting confidence. Test on the belt
before treating any prediction as a sorting decision.

## External dataset check (2026-09-23)

The unchanged checkpoint was evaluated on a reproducible sample of
[Alistair King's Recyclable and Household Waste Classification dataset](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification).
Run `python -m material_classifier.evaluate_kaggle_household --archive
material_classifier/data/recyclable-household.zip` after downloading the
dataset archive to that ignored path. The evaluator samples 25 images from
each of 30 item categories in each of the `default` and `real_world` folders,
using seed 2026. It writes `summary.json` and per-image `predictions.csv` to
`material_classifier/output/kaggle-household/`.

Only 21 item categories have a clear match to the model's material labels.
On those 1,050 sampled images, top-1 accuracy was **47.1%** (495 correct).
The `default` images scored 48.2% and `real_world` images 46.1%. Recall by
mapped material was cardboard 85.0%, paper 81.3%, metal 55.5%, plastic
32.4%, and glass 20.7%. The other 450 sampled images, including organics,
textiles, foam, and paper cups, were excluded from accuracy because the model
has no directly matching class or the material is ambiguous. This dataset
contains staged scenes and images with multiple objects, so these results are
an external generalization check, not a conveyor accuracy estimate.

## First live belt check (2026-09-22)

The Mac's Brio 100 camera successfully fed the local preview with a disposable
cup on the small conveyor. A close crop of one frame predicted `cardboard` at
98.5%, while a later camera position gave `paper` as the top result in 20 of
20 consecutive frames (mean top score 73.6%). The camera and cup framing
changed between these runs. The cup's actual material was not established, so
these observations measure prediction stability and sensitivity to framing,
not correctness. In the side view, the top of the cup was sometimes outside
the frame and people and equipment filled much of the background. The next
useful evaluation is a fixed overhead view and labeled photos of known belt
items, with physical items separated between training and test sets.

## Three belt photos: preprocessing check (2026-09-23)

On user-labeled cup, snack wrapper, and plastic bottle photos, the deployed
centered 90% square crop predicted `paper` (65.0%), `cardboard` (96.5%), and
`paper` (91.0%) respectively. A single fixed 550-pixel crop around the belt
made the cup and bottle material predictions acceptable, but the wrapper
remained `cardboard`.

Supplying an item rectangle for each photo, running OpenCV GrabCut within a
60-pixel margin, compositing the foreground onto white, and padding to a square
before the usual model transform produced `cardboard` for the cup (94.6%),
`trash` for the wrapper (66.8%), and `plastic` for the bottle (68.8%). The
processed images and full scores are in the ignored
`material_classifier/output/belt_preprocessing/` directory. This is a
diagnostic experiment: the rectangles were selected manually. A shared fixed
rectangle failed, and the existing YOLOX Nano detector did not localize the
cup or wrapper in these photos. Live use requires reliable object localization
and more labeled belt images before these results can be generalized.

## UNO Q export

The seven-label checkpoint is exported to ONNX in `uno_q_material_logger/model/`.
The standalone App Lab app uses ONNX Runtime on the UNO Q's Linux processor,
with the Mac preview's input crop, resize, normalization, and label order.
The repository parity test compares its probabilities against the PyTorch
checkpoint. This seven-label export has not been deployed to a UNO Q in this
training run, so board behavior remains to be checked there.
