# Wrapper false positives: focused model experiment

The previous seven-class checkpoint calls the supplied snack wrapper
`cardboard` (46.6%) on the default 0.9 square ROI. A tight manual crop still
calls it `cardboard` (94.6%), so changing the ROI alone cannot solve this
case. The broad ROI also includes a white paper sheet and other foreground
objects; a tight bottle crop changes that bottle's top prediction from
`paper` to `glass`, confirming a separate framing problem.

## Data and split

Source: [TACO official dataset](https://github.com/pedropro/TACO), using its
official `data/annotations.json` and Flickr 640px URLs. Only the annotated
`Crisp packet` and `Other plastic wrapper` categories map to our `trash`
material label. `Plastic film` is excluded because that category alone does
not establish the local recycling rule. Crops use the annotated bounding box
with 12% padding. Objects whose projected short side is under 40px are
excluded.

`prepare_taco_wrappers.py` selects one largest qualifying annotation per
source image. It shuffles source image IDs with seed 42 and reserves 25% of
those IDs before download or crop. The resulting 58 train crops are added to
the existing mixed-data **training** split only; 20 crops stay in a separate
`taco_test` folder and are absent from training, validation, and the mixed
test split. Existing mixed split files are linked into an ignored local
directory, preserving their original assignments. The ignored
`taco_manifest.json` records each image ID, category, source URL and split.
Downloaded images and TACO crops are not committed.
The 20 reserved crops also have no close train duplicate at dHash Hamming
distance 2 or less.

Recreate from repository root, with the preexisting mixed dataset available:

```sh
.venv/bin/python -m material_classifier.prepare_taco_wrappers \
  --source-dir material_classifier/data/food-organic-mixed \
  --output-dir material_classifier/data/taco-wrapper-experiment
```

## Baseline

The previous checkpoint at commit `ff38f50` gets 4/20 of the reserved
TACO wrapper crops right as `trash`. It calls 7 `food_organic`, 3
`cardboard`, 3 `paper`, and 3 `plastic`. On the supplied belt images with
the preview's default 0.9 crop: cup=`paper` 97.0%, wrapper=`cardboard`
46.6%, bottle=`paper` 58.1%. The known labels are cup=`paper` or
`cardboard`, wrapper=`trash`, bottle=`plastic`.

## Training and results

Continue the prior seven-label checkpoint, keeping the original mixed test
set untouched:

```sh
git show ff38f50996c83bfd5032061533444350f3608497:material_classifier/models/food-organic-mobilenetv3-small.pt \
  > /tmp/food-organic-before-wrapper.pt
.venv/bin/python -m material_classifier.train \
  --data-dir material_classifier/data/taco-wrapper-experiment/mixed \
  --output-dir material_classifier/output/taco-wrapper-v1 \
  --start-from /tmp/food-organic-before-wrapper.pt \
  --balance-classes --selection-metric macro_recall \
  --train-samples-per-epoch 3500 --epochs 4 --batch-size 32 \
  --learning-rate 0.00005 --unfreeze-last 2
```

The checkpoint selected after epoch 2 is committed as
`models/food-organic-mobilenetv3-small.pt`; the full held-out confusion
matrix is in its adjacent `.metrics.json`. Use a fresh copy of the **prior**
checkpoint as `--start-from` when repeating this command; the `git show`
line above extracts it from the parent commit.

| Check | Before | After |
| --- | ---: | ---: |
| Reserved TACO wrapper crops predicted `trash` | 4/20 | **18/20** |
| Original mixed test accuracy | 916/1,118 (81.9%) | **931/1,118 (83.3%)** |
| Original mixed test macro recall | 81.4% | **83.3%** |
| Original mixed test `trash` recall | 77/96 (80.2%) | **79/96 (82.3%)** |
| Original mixed test `food_organic` recall | 183/194 (94.3%) | **184/194 (94.8%)** |
| Reserved RealWaste accuracy | 472/603 (78.3%) | **483/603 (80.1%)** |
| Reserved RealWaste `trash` recall | 61/75 (81.3%) | **64/75 (85.3%)** |
| Reserved RealWaste `food_organic` recall | 53/62 (85.5%) | **54/62 (87.1%)** |

In the RealWaste subset, metal, paper, and plastic each lost 1, 1, and 3
correct predictions; cardboard, food, glass, and trash gained 9, 1, 3, and 3.
The model still calls the supplied belt wrapper `cardboard` with 68.7%
confidence (up from 46.6%) and the bottle `paper` with 69.8% (up from 58.1%).
The cup remains `paper` (94.9%). A manual white-background foreground of the
wrapper, cropped to the central 90%, becomes `trash` at 45.9%; the raw tight
wrapper crop remains `cardboard` at 96.4%. Thus, this update improves
independent wrapper recognition but does not establish conveyor accuracy.
The next needed data are several distinct wrapper products photographed in
the fixed belt camera view, kept separate by physical item across train/test.

Recheck the reserved TACO examples and three belt photos with:

```sh
.venv/bin/python -m material_classifier.evaluate_wrappers \
  --checkpoint material_classifier/models/food-organic-mobilenetv3-small.pt \
  --test-dir material_classifier/data/taco-wrapper-experiment/taco_test \
  --belt-images /path/to/cup_conv_belt.jpg \
                /path/to/wrapper_conv_belt.jpg \
                /path/to/bottle_conv_belt.jpg
```
