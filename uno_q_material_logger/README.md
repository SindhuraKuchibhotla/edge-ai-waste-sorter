# UNO Q material classifier logger

This App Lab app runs the repository's seven-label MobileNetV3 Small model
on the UNO Q's Linux processor. It reads a USB camera once per second, saves
the top material label, score, and disposal lane to persistent text files, and
sends each prediction to the OLED sketch over RouterBridge. No laptop, browser, Edge
Impulse service, or network is needed while it runs. It does not draw
bounding boxes or operate a sorter.

## What gets recorded

The app creates a new `data/results-YYYY-MM-DDTHH-MM-SS.ffffffZ.txt` file each
time it starts, using the board's UTC clock. It keeps appending one JSON object
per line to that file until the app stops, even if a run crosses midnight.
Previous files remain on the board until you delete them. Each line has
`timestamp_utc`, `uptime_s`, `camera_status`,
`material`, `confidence`, `lane`, `inference_ms`, and `error`. `camera_status=ok` means a usable camera
frame was received; `unavailable` means the camera could not open; `read_error`
means an open camera stopped returning frames. When capture fails, material and
confidence and lane are null. The app retries the camera every five seconds.
The OLED shows the lane, material, and confidence for successful predictions. Its
on-board RGB LED #3 shows green for compost, blue for recycle, red for trash,
matching the three disposal lanes. The external LEDs use the pin map from
`origin/sk/integration`: D3 green for compost, D5 blue for recycle, D6 red
for trash, D9 yellow for a score below 60%, and D12 white while
idle or when the camera is unavailable. Wire each output through a
current-limiting resistor to the LED anode, with its cathode at GND. These
external outputs are active high. Camera or model errors turn built-in LED #3
off and light the white external LED. The OLED
shows `NO CAMERA` or `MODEL ERROR` when a new prediction cannot be made.
Display communication errors do not stop the results file.
The package includes Adafruit SSD1306 2.5.17 with one UNO Q Zephyr
compatibility change: its unused software-SPI port optimization is disabled
for Zephyr. Adafruit GFX 1.12.6 and BusIO 1.17.3 are also bundled so the OLED
sketch can build without downloading libraries. Their upstream licenses are
included with the sources.

An example line:

```json
{"timestamp_utc":"2026-09-22T16:00:00Z","uptime_s":20.004,"camera_status":"ok","material":"paper","confidence":0.736,"lane":"recycle","inference_ms":38,"error":null}
```

Labels are `cardboard`, `food_organic`, `glass`, `metal`, `paper`, `plastic`,
and `trash`. `plastic` and `trash` map to the trash lane, `food_organic` to
compost, and cardboard, glass, metal, and paper to recycle. A low model score
does not change the lane: below 60%, the yellow LED lights and the OLED flashes.
An unknown material label maps to trash. A non-finite score is recorded as a
model error with no lane.
The OLED shows the lane in large text and keeps the full material label beneath it.
The model always chooses one of these labels when a frame arrives, even if the
belt is empty. The score is an uncalibrated model probability, not proof of the
material or the correct disposal lane.

## Install once, then run without the laptop

1. With the camera disconnected, attach the UNO Q to your laptop by USB-C and
   open Arduino App Lab. Import `dist/material-classifier-oled-uno-q.zip` using
   **My Apps → Import App**. The ZIP includes the model, sketch, and OLED
   libraries. Keep the board online for the first run so App Lab can install
   the Python dependencies.
2. Select **Material Classifier OLED** and press **Run**. A missing-camera
   record every second is expected during this setup. The first run may take
   longer while App Lab installs the Python packages.
3. Enable **Run at startup** for this app in App Lab's project-name menu.
   This is a separate setting from pressing Run once.
4. Wire the 128×64 SSD1306 OLED to the UNO Q's I²C pins: GND to GND, VCC to
   3V3, SCL to SCL, and SDA to SDA. The sketch uses address `0x3C`. Power
   down. Connect the camera to a powered USB-C hub, connect the hub to the UNO
   Q, and power the hub. Keep the camera attached before powering up the board.
   The app should start on its own after Linux boots.
5. For a short test, leave the board running with one known item in view. Then
   power down, remove the camera hub, and reconnect the laptop by USB-C.
   In App Lab's board terminal, find the file under the imported app's
   `data/` directory. App Lab normally stores user apps below
   `/home/arduino/ArduinoApps/`; this command locates the files regardless of
   the app folder name:

   ```sh
   find /home/arduino/ArduinoApps -path '*/data/results-*.txt' -print
   ```

   Use `cat` or `tail` on the path printed above to read and copy the text.
   If the board clock has not synchronized, the filename time may be wrong;
   `uptime_s` still gives the order within each run.

## Camera framing

The camera requests 640×480 frames. `config.json` controls the square region
used for classification, with the same fractional crop controls as the Mac
preview. Its defaults select a centered region covering 90% of the frame.
Mount the camera so one item occupies that region. After testing, edit the
four values in `config.json`, rebuild the ZIP, and import the updated app if
the item is off center. The app does not store camera images.

## Rebuild the package

The committed ONNX file was exported from
`material_classifier/models/food-organic-mobilenetv3-small.pt`. After training
a newer checkpoint, run these commands in the repository's Python environment:

```sh
python -m pip install -r material_classifier/export-requirements.txt
python -m material_classifier.export_onnx --checkpoint path/to/new-best.pt
python -m material_classifier.package_uno_q
```

The exporter uses the training environment's `torch` and `torchvision`.
The UNO Q app requires only ONNX Runtime, Pillow, and their dependencies;
it does not install PyTorch.

## Empty-belt check

The model labels every frame, even an empty belt, so the app first checks whether
anything is on the belt. For the first 3 seconds after it starts it learns what the
empty belt looks like, so start it with the belt empty. After that it sends a result
to the OLED only while something is on the belt: `display_scanning` when an item
appears, then `show_result` every second until it leaves. Result lines also record
`item_present` and `difference`. If items are missed or the empty belt counts as an
item, set `presence_threshold` in `config.json` between the empty and item values
(default 12). Set `detect_empty_belt` to `false` to send every result as before.
