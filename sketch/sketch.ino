// =====================================================================
//  THE GARBAGE COLLECTORS - display, LEDs and belt (Arduino UNO Q, STM32 side)
// =====================================================================
//
//  FILES (App Lab / the IDE join every .ino in this folder into one sketch,
//  sketch.ino first, the rest in alphabetical order)
//   sketch.ino       settings, pins, shared state, setup(), loop()
//   bridge_api.ino   the functions Python calls + turning them into a result
//   screens.ino      text helpers, boot screen, RESULT / STATS / SYSTEM pages, scanner
//   animations.ino   ~1 s animation per lane
//   leds.ino         lane LEDs, yellow "human check" LED, white idle/scan LED
//   belt.ino         motor control (only does something with MOTOR_MODE 1 or 2)
//   libraries/       Adafruit SSD1306 (UNO Q Zephyr fix), GFX and BusIO, bundled (sketch.yaml)
//
//  FLOW
//   BOOT      truck drives across -> title types out -> loading bar + LED chase -> tagline
//   IDLE      scanner "WAITING FOR ITEM" (8 s) <-> stats peek (4 s, once items exist)
//   SCANNING  Python calls display_scanning when an item shows up
//   RESULT    ~1 s lane animation, then RESULT 4 s -> STATS 3 s -> SYSTEM 3 s -> ...
//             20 s without a new item -> back to IDLE
//   RAPID FIRE turns on by itself when two results arrive less than 4 s apart
//             (AUTO_RAPID), or always with RAPID_FIRE 1 / Python set_rapid_fire(1).
//             No SCANNING screen, 1 s animation, the RESULT page stays (header
//             "RAPID FIRE x<streak>") and is replaced by the next item. 6 s with no
//             item -> STATS page and rapid fire ends; 20 s -> IDLE
//   BELT TIMING (measured): a small item takes ~2 s end to end, with or without
//             the 2x10 ohm resistors. Python decides ~0.75 s after an item enters and
//             needs ~0.75 s of empty belt before the next one, so back-to-back items
//             give a result every ~2.75 s (never faster). The rapid fire times below
//             follow from that; change them if the belt speed changes.
//   Any animation stops early when the next result arrives, so results never queue up.
//   LOW SCORE (< CONF_MIN) "HUMAN CHECK NEEDED" header, result page flashes,
//             yellow LED breathes, best-guess lane LED stays dim
//
//  PYTHON CALLS (App Lab)
//   Material classifier app (uno_q_material_logger), about once per second:
//   Bridge.notify("show_result", "RECYCLE", "cardboard", 85, 38)  # lane, material, score %, ms
//   Bridge.notify("show_status", "NO CAMERA")                      # or "MODEL ERROR"
//   Bridge.notify("display_scanning")                              # new item on the belt
//     With the empty-belt check on, Python sends display_scanning when an item arrives,
//     then show_result every second while it is in view, and nothing while the belt is
//     empty. Everything between two display_scanning calls is one item; if a reading
//     of another lane is surer than the one on screen, the item is re-labelled. Without the check, repeats of the same lane update the item
//     and a new lane must arrive STREAM_CONFIRM times in a row before it counts.
//     Python maps plastic and trash to TRASH, food_organic to COMPOST, the rest to
//     RECYCLE; an unknown or empty lane is worked out from the material. CONF_MIN
//     decides HUMAN CHECK.
//   One call per item (other Python apps):
//   Bridge.call("display_item", "RECYCLE", "bottle", 85, 38)  # lane, what it is, score %, ms
//   Bridge.call("display_result", "cardboard", 85, 38)        # material or lane, score %, ms
//   Bridge.call("display_scanning")                           # item seen, working on it
//   Bridge.call("run_conveyor", "RECYCLE")                    # carry the item off
//   Bridge.call("reset_stats")
//   Bridge.call("set_rapid_fire", 1)                          # 1 = rapid fire on, 0 = off
//   Also: display_label(text), display_score(%), display_time(ms),
//         belt_on(), belt_stop(), belt_speed(0..255), belt_run_for(ms)
//   Use -1 for a score or time you don't have; the screen shows "--".
//
//  WIRING
//   OLED  : GND->GND, VCC->3V3, SCL->SCL, SDA->SDA (I2C address 0x3C)
//   LEDs  : pin -> resistor -> LED long leg, short leg -> GND rail
//           D3 green COMPOST, D5 blue RECYCLE, D6 red TRASH, D9 yellow LOW SCORE, D12 white IDLE
//   Motor : MOTOR_MODE 0 -> motor on its resistors, code can't control it (current setup)
//           MOTOR_MODE 1 -> NPN (2N2222): D10 -> 1k -> base, emitter -> GND,
//                           collector -> motor (-), motor (+) -> 3V3, diode across the motor
//           MOTOR_MODE 2 -> driver board (DRV8833 / L9110S): D10 -> IN1, D11 -> IN2
// =====================================================================

#include <Wire.h>
#include <math.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>      // bundled in libraries/ (has the Zephyr fix)
#include <Arduino_RouterBridge.h>

// ---------------- settings ----------------
#define DEMO_MODE   0     // 1 = fake items (test without Python), 0 = real results from Python
#define MOTOR_MODE  0     // 0 = no motor control, 1 = NPN transistor, 2 = driver board
#define BELT_SPEED  255   // 0..255. With a transistor and NO diode keep 255 (plain on/off)
#define CONF_MIN    60    // score below this % = low score / human check
#define RAPID_FIRE  0     // 1 = always rapid fire (see FLOW above)
#define AUTO_RAPID  1     // 1 = rapid fire switches on by itself when items come fast
#define STREAM_CONFIRM 2  // show_result: readings of a new lane in a row before it counts
                          // (2 stops one-frame flicker; 1 = count at once, for fast belts)

// timings in milliseconds
const unsigned long T_RESULT       = 4000;
const unsigned long T_STATS        = 3000;
const unsigned long T_SYSTEM       = 3000;
const unsigned long T_IDLE_SCAN    = 8000;
const unsigned long T_IDLE_STATS   = 4000;
const unsigned long T_BACK_TO_IDLE = 20000;
const unsigned long T_SCAN_TIMEOUT = 10000;
const unsigned long T_FLASH        = 400;
const unsigned long T_CONVEYOR     = 1500;   // run_conveyor(): how long the belt runs
const unsigned long T_RAPID_TRIGGER = 4000;  // two results closer than this -> rapid fire
const unsigned long T_RAPID_PAUSE  = 6000;   // rapid fire: no item this long -> STATS, ends
const unsigned long T_ANIM_RAPID   = 1000;   // rapid fire animation length
const unsigned long T_ANIM_NORMAL  = 1000;   // normal animation length
const unsigned long T_SAME_ITEM    = 3000;   // show_result: same lane within this = same item

// pins
#define LED_GREEN   3     // PWM
#define LED_BLUE    5     // PWM
#define LED_RED     6     // PWM
#define LED_YELLOW  9     // PWM
#define LED_WHITE   12    // on/off only
#define MOTOR_A     10    // PWM: transistor base (via 1k) or driver IN1
#define MOTOR_B     11    // driver IN2 (mode 2 only)

#define WHT SSD1306_WHITE
#define BLK SSD1306_BLACK

Adafruit_SSD1306 display(128, 64, &Wire, -1);

const char* LANES[3]    = {"COMPOST", "RECYCLE", "TRASH"};
const char* SHORTS[3]   = {"COMP", "RECY", "TRASH"};
const int   LANE_LED[3] = {LED_GREEN, LED_BLUE, LED_RED};

// ---------------- incoming from Python (set by the Bridge functions) ----------------
volatile bool inNewItem = false, inDirty = false, inScan = false, inReset = false;
String inText = "";            // lane or material
String inObject = "";          // what the item is (display_item / show_result)
volatile bool inStream = false;          // result came from show_result (1 per second)
volatile bool inStatus = false;          // show_status arrived
volatile bool inItemStart = false;       // display_scanning: the next show_result is a new item
bool itemOpen = false;                   // current item started with display_scanning
String inStatusText = "";
volatile int inScore = -1, inMs = -1;
volatile unsigned long inChangedAt = 0;

// ---------------- current item ----------------
String curLabel = "", curMaterial = "";
int  curLane = -1, curScore = -1, curMs = -1;
bool curUnsure = false, curMsCounted = false, curScoreCounted = false;

// ---------------- stats ----------------
int confCnt[3] = {0, 0, 0}, unsureCnt[3] = {0, 0, 0}, total = 0;
unsigned long sumMs = 0, sumScore = 0;
int msN = 0, scoreN = 0, fastestMs = -1;

// ---------------- screen state ----------------
enum Mode { M_IDLE, M_SCANNING, M_RESULT, M_STATUS };
Mode mode = M_IDLE;
int page = 0, idleSub = 0, frame = 0;
unsigned long pageAt = 0, modeAt = 0, frameAt = 0, lastItemAt = 0, flashAt = 0;
unsigned long lastSeenAt = 0;      // last result of any kind (new item or update)
int pendingLane = -1;              // show_result: new lane waiting for STREAM_CONFIRM readings
int pendingCount = 0;
unsigned long pendingAt = 0;
bool inverted = false, booting = true;

// ---------------- rapid fire ----------------
bool rapidForced = RAPID_FIRE;     // always on: RAPID_FIRE 1 or Python set_rapid_fire(1)
bool rapidAuto = false;            // switched on by AUTO_RAPID during a burst
bool rapidStats = false;           // true once the pause STATS page is showing
int  streak = 0;                   // items in the current burst

// ---------------- LEDs ----------------
int ledLevel[3] = {0, 0, 0}, ledTarget[3] = {0, 0, 0};
bool yellowBreath = false;
unsigned long ledStepAt = 0;

// ---------------- belt ----------------
int beltSpeed = BELT_SPEED;
bool beltOn = false;
unsigned long beltStopAt = 0;

// =====================================================================
//  setup / loop
// =====================================================================
void setup() {
  Bridge.begin();
  Monitor.begin(9600);
  Wire.begin();
  Monitor.println("setup: bridge ok");

  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_BLUE, OUTPUT);
  pinMode(LED_RED, OUTPUT);
  pinMode(LED_YELLOW, OUTPUT);
  pinMode(LED_WHITE, OUTPUT);
#if defined(LED3_R)
  pinMode(LED3_R, OUTPUT);           // UNO Q on-board RGB LED #3 (active low)
  pinMode(LED3_G, OUTPUT);
  pinMode(LED3_B, OUTPUT);
#endif
#if MOTOR_MODE >= 1
  pinMode(MOTOR_A, OUTPUT);
#endif
#if MOTOR_MODE == 2
  pinMode(MOTOR_B, OUTPUT);
#endif
  motorApply();                      // belt starts stopped

  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    while (true) { Monitor.println("OLED failed to start (check wiring, address 0x3C)"); delay(2000); }
  }
  display.clearDisplay();            // something on screen straight away
  display.setTextColor(WHT);
  display.setTextSize(1);
  display.setCursor(0, 28);
  display.print("starting...");
  display.display();
  Monitor.println("setup: oled ok");

  provideBridgeFunctions();          // bridge_api.ino
  Monitor.println("setup: bridge functions ok");
  bootScreen();                      // screens.ino
  goIdle();
  Monitor.println("setup: done, waiting for items");
}

void loop() {
  unsigned long now = millis();

#if DEMO_MODE
  runDemo(now);                      // bridge_api.ino
#endif

  serviceBelt();

  if (inReset) {
    inReset = false;
    for (int i = 0; i < 3; i++) { confCnt[i] = 0; unsureCnt[i] = 0; }
    total = msN = scoreN = 0; sumMs = sumScore = 0; fastestMs = -1;
  }

  if (inScan) {                                   // item under the camera
    inScan = false;
    if (!rapidNow()) {                            // rapid fire skips the SCANNING screen
      mode = M_SCANNING;
      modeAt = now;
      frame = 0;
      inverted = false;
      display.invertDisplay(false);
      setLedsOff();
    }
  }

  if (inStatus) {                                 // NO CAMERA / MODEL ERROR from Python
    inStatus = false;
    showStatus(inStatusText);                     // screens.ino
  }

  if (inDirty && now - inChangedAt > 40) {        // wait for all calls of one item
    commitIncoming();
    now = millis();
  }

  switch (mode) {
    case M_IDLE:
      if (idleSub == 0 && total > 0 && now - modeAt > T_IDLE_SCAN) {
        idleSub = 1; modeAt = now; drawStatsPage();
      } else if (idleSub == 1 && now - modeAt > T_IDLE_STATS) {
        idleSub = 0; modeAt = now;
      }
      if (idleSub == 0 && now - frameAt > 50) {
        frameAt = now;
        drawScanner(frame++, "WAITING FOR ITEM", 1);
      }
      break;

    case M_SCANNING:
      if (now - modeAt > T_SCAN_TIMEOUT) {
        goIdle();
      } else if (now - frameAt > 40) {
        frameAt = now;
        drawScanner(frame++, "SCANNING", 2);
      }
      break;

    case M_RESULT: {
      if (now - lastSeenAt > T_BACK_TO_IDLE) { goIdle(); break; }
      if (rapidNow() || rapidStats) {
        // RESULT stays up during a burst; after a pause show STATS until the next item
        if (!rapidStats && now - lastItemAt > T_RAPID_PAUSE) {
          rapidStats = true;
          rapidAuto = false;               // burst over
          streak = 0;
          page = 1;
          inverted = false;
          display.invertDisplay(false);
          setLedsOff();
          drawStatsPage();
        }
        if (!rapidStats && curUnsure && now - flashAt > T_FLASH) {  // low score: flash
          flashAt = now;
          inverted = !inverted;
          display.invertDisplay(inverted);
        }
        break;
      }
      unsigned long dur = (page == 0) ? T_RESULT : (page == 1 ? T_STATS : T_SYSTEM);
      if (now - pageAt > dur) {
        page = (page + 1) % 3;
        pageAt = now;
        drawPage();
      }
      if (page == 0 && curUnsure && now - flashAt > T_FLASH) {   // low score: flash
        flashAt = now;
        inverted = !inverted;
        display.invertDisplay(inverted);
      }
      break;
    }

    case M_STATUS:                                // stays until the next result
      break;
  }

  serviceLeds();
  delay(5);
}
