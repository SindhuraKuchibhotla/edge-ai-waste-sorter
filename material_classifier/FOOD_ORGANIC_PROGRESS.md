# Food-organic training record (2026-09-23)

Goal: add `food_organic` to the existing six-material MobileNetV3 Small model.
The trained checkpoint is `models/food-organic-mobilenetv3-small.pt`. The Mac
preview now selects it by default, and the UNO Q app's bundled ONNX file is
exported from it. This new version has not been deployed to the physical board.

## Training path

- `models/food-organic-mixed-paused.pt` preserves the intermediate checkpoint
  from an interrupted mixed-data run, after epoch 2 of 6. It has seven labels.
  Its SHA-256 is `063166cfac44aaa5f4baf80f50fa1bb6c182077d8b9dbf9ece904ca96bceeb88`.
- The same intermediate checkpoint remains at `output/food-organic-v4/best.pt`.
  The optimizer and sampler positions were not saved, so the final four-epoch
  run restarted its optimizer from these weights.
- Validation after epoch 1: accuracy 73.6%, macro recall 72.3%. After epoch 2:
  accuracy 77.0%, macro recall 76.1% (best saved checkpoint).
- `data/food-organic-mixed/source_manifest.json` records the dataset
  sources and train/validation/test assignment. The images and source archives
  are in ignored local `data/` folders. The mixed splits have 5,167 training,
  1,107 validation, and 1,118 test images. The RealWaste part of the test
  split has 603 images.

## How we got here

`prepare_food_organic.py` combined the original TrashNet split with the
`food_waste`, `eggshells`, and `coffee_grounds` folders of King's Recyclable
and Household Waste dataset. It removed near-duplicate organic images before
splitting, leaving 869 distinct photos. Tea bags were excluded. The first
seven-class checkpoint is `output/food-organic-v3/best.pt`; its in-dataset
food-organic recall was 131/132, but on independent UCI RealWaste photos its
food-organic recall was 314/411 with only 43.4% precision. It also mislabeled
the belt bottle photo as organic. It should not be promoted.

`prepare_food_organic_mixed.py` added deduplicated UCI RealWaste examples from
all seven matching classes to the existing splits. The original six output
rows were preserved when the seventh class was added. The mixed-data phase
unfroze the final two backbone blocks and sampled classes equally.

## Final four-epoch run

The completed run used the following command from the repository root. Use a
fresh output directory to repeat it; the dataset archives are not committed.

```sh
.venv/bin/python -m material_classifier.train \
  --data-dir material_classifier/data/food-organic-mixed \
  --output-dir material_classifier/output/food-organic-v4-repeat \
  --start-from material_classifier/models/food-organic-mixed-paused.pt \
  --balance-classes --selection-metric macro_recall \
  --train-samples-per-epoch 3500 --epochs 4 --batch-size 32 \
  --learning-rate 0.0001 --unfreeze-last 2
```

The recorded best checkpoint is at `output/food-organic-v4-resumed/best.pt`.
It was exported to `models/food-organic-mobilenetv3-small.pt`. Evaluate only
the reserved RealWaste test images with:

```sh
.venv/bin/python -m material_classifier.evaluate_realwaste \
  --archive material_classifier/data/realwaste.zip \
  --checkpoint material_classifier/output/food-organic-v4-resumed/best.pt \
  --source-manifest material_classifier/data/food-organic-mixed/source_manifest.json \
  --output material_classifier/output/food-organic-v4-resumed/realwaste-test
```

On the mixed held-out test set, accuracy was 916/1,118 (81.9%), macro recall
81.4%, food-organic recall 183/194 (94.3%), and food-organic precision 183/192
(95.3%). On the reserved RealWaste test paths, accuracy was 472/603 (78.3%),
macro recall 78.8%, food-organic recall 53/62 (85.5%), and food-organic
precision 53/57 (93.0%). The first seven-class checkpoint scored 41.5%
accuracy and 43.8% food-organic precision on those same 603 RealWaste paths.

On the three supplied nonorganic belt photos, the new model did not select
`food_organic` as its top class. It still misclassified the wrapper and bottle
materials with the current crop. No belt food-organic photo was available for
evaluation. Its score is not calibrated and an empty view still receives a
material label. More fixed-camera belt images are needed before using the
new class for sorting decisions.
