// =====================================================================
//  leds.ino - lane LEDs fade, yellow breathes on a low score,
//  white is steady in idle and blinks while scanning
// =====================================================================

// UNO Q on-board RGB LED #3 (active low): green compost, blue recycle, red trash,
// amber when unsure, off otherwise. Skipped on boards without it.
void setOnboardRgb(int lane, bool unsure) {
#if defined(LED3_R)
  bool r = lane >= 0 && (lane == 2 || unsure);
  bool g = lane >= 0 && (lane == 0 || unsure);
  bool b = lane == 1 && !unsure;
  digitalWrite(LED3_R, r ? LOW : HIGH);
  digitalWrite(LED3_G, g ? LOW : HIGH);
  digitalWrite(LED3_B, b ? LOW : HIGH);
#endif
}

void setLedsForResult() {
  for (int i = 0; i < 3; i++)
    ledTarget[i] = (i == curLane) ? (curUnsure ? 40 : 255) : 0;
  yellowBreath = curUnsure;
  setOnboardRgb(curLane, curUnsure);
}

void setLedsOff() {
  for (int i = 0; i < 3; i++) ledTarget[i] = 0;
  yellowBreath = false;
  setOnboardRgb(-1, false);
}

void serviceLeds() {
  if (booting) return;                        // boot sequence drives the LEDs itself
  unsigned long now = millis();
  if (now - ledStepAt >= 15) {                // smooth fade towards target
    ledStepAt = now;
    for (int i = 0; i < 3; i++) {
      if (ledLevel[i] < ledTarget[i]) ledLevel[i] = (ledLevel[i] + 12 > ledTarget[i]) ? ledTarget[i] : ledLevel[i] + 12;
      if (ledLevel[i] > ledTarget[i]) ledLevel[i] = (ledLevel[i] - 12 < ledTarget[i]) ? ledTarget[i] : ledLevel[i] - 12;
      analogWrite(LANE_LED[i], ledLevel[i]);
    }
    if (yellowBreath) {                       // slow breathing, 2 s per breath
      float s = (sinf(now * 0.00314f) + 1.0f) * 0.5f;
      analogWrite(LED_YELLOW, (int)(s * s * 255));
    } else {
      analogWrite(LED_YELLOW, 0);
    }
  }
  // white: steady while idle or when the camera/model is down, blinks while scanning
  bool white = (mode == M_IDLE) || (mode == M_STATUS) || (mode == M_SCANNING && (now / 125) % 2 == 0);
  digitalWrite(LED_WHITE, white ? HIGH : LOW);
}

// like delay(), but keeps LEDs fading and the belt timer running
void pause(unsigned long ms) {
  unsigned long t = millis();
  while (millis() - t < ms) { serviceBelt(); serviceLeds(); delay(2); }
}
