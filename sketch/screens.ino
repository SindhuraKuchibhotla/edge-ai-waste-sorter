// =====================================================================
//  screens.ino - text helpers, boot screen, pages, idle/scan scanner
// =====================================================================

// ---------------- text helpers ----------------
void centerText(const String& s, int y, int size) {
  display.setTextSize(size);
  display.setCursor((128 - (int)s.length() * 6 * size) / 2, y);
  display.print(s);
}

void rightText(const String& s, int y) {
  display.setTextSize(1);
  display.setCursor(128 - (int)s.length() * 6, y);
  display.print(s);
}

void row(const char* name, const String& value, int y) {
  display.setTextSize(1);
  display.setCursor(0, y);
  display.print(name);
  rightText(value, y);
}

// value + unit, or "--" when not known yet
String withUnit(long v, const char* unit) {
  if (v < 0) return String("--");
  String out = String(v);
  out += unit;
  return out;
}

// ---------------- boot ----------------
void drawTruck(int x, int y) {
  display.drawRect(x, y, 36, 18, WHT);
  for (int i = 6; i < 36; i += 6) display.drawFastVLine(x + i, y + 2, 14, WHT);
  display.fillRect(x + 36, y + 6, 14, 12, WHT);
  display.fillRect(x + 39, y + 8, 7, 5, BLK);
  display.fillCircle(x + 9,  y + 20, 4, WHT);
  display.fillCircle(x + 41, y + 20, 4, WHT);
  display.fillCircle(x + 9,  y + 20, 1, BLK);
  display.fillCircle(x + 41, y + 20, 1, BLK);
}

void bootScreen() {
  booting = true;

  // 1. truck drives across (~1.5 s)
  for (int x = -52; x <= 128; x += 5) {
    display.clearDisplay();
    display.setTextColor(WHT);
    centerText("powering on...", 4, 1);
    display.drawFastHLine(0, 55, 128, WHT);
    drawTruck(x, 30);
    if ((x / 5) % 2 == 0) display.drawCircle(x - 5, 46, 2, WHT);
    display.display();
    delay(35);
  }

  // 2. title types out (~1.2 s)
  const char* l1 = "GARBAGE";
  const char* l2 = "COLLECTORS";
  char buf[12];
  for (int i = 1; i <= 17; i++) {
    display.clearDisplay();
    centerText("THE", 0, 1);
    display.setTextSize(2);
    int n1 = i < 7 ? i : 7;
    strncpy(buf, l1, n1); buf[n1] = 0;
    display.setCursor((128 - 7 * 12) / 2, 12);
    display.print(buf);
    if (i > 7) {
      int n2 = i - 7;
      strncpy(buf, l2, n2); buf[n2] = 0;
      display.setCursor((128 - 10 * 12) / 2, 32);
      display.print(buf);
    }
    display.display();
    delay(70);
  }

  // 3. loading bar + LED chase W Y G B R (~1 s)
  const int chase[5] = {LED_WHITE, LED_YELLOW, LED_GREEN, LED_BLUE, LED_RED};
  for (int s = 0; s < 50; s++) {
    int idx = s / 10, phase = s % 10;
    int b = (phase < 5) ? phase * 60 : (9 - phase) * 60;     // fade up then down
    for (int j = 0; j < 5; j++) {
      int v = (j == idx) ? clampi(b, 0, 255) : 0;
      if (chase[j] == LED_WHITE) digitalWrite(LED_WHITE, v > 100 ? HIGH : LOW);
      else analogWrite(chase[j], v);
    }
    display.fillRect(0, 52, 128, 12, BLK);
    display.drawRect(14, 54, 100, 8, WHT);
    display.fillRect(16, 56, (96 * (s + 1)) / 50, 4, WHT);
    display.display();
    delay(20);
  }
  for (int j = 0; j < 5; j++) {
    if (chase[j] == LED_WHITE) digitalWrite(LED_WHITE, LOW); else analogWrite(chase[j], 0);
  }

  // 4. tagline (~1.3 s)
  display.fillRect(0, 52, 128, 12, BLK);
  centerText("edge AI waste sorter", 54, 1);
  display.display();
  delay(1300);

  booting = false;
}

// ---------------- pages ----------------
void drawHeader(bool human) {
  if (human) {                                  // inverted warning bar
    display.fillRect(0, 0, 128, 11, WHT);
    display.setTextColor(BLK);
    centerText("HUMAN CHECK NEEDED", 2, 1);
    display.setTextColor(WHT);
  } else {
    if (rapidNow()) centerText("RAPID FIRE x" + String(streak), 0, 1);
    else           centerText("GARBAGE COLLECTORS", 0, 1);
    display.drawFastHLine(0, 10, 128, WHT);
  }
}

// RESULT: lane, what it is, score bar + %, inference time
void drawResultPage() {
  display.clearDisplay();
  display.setTextColor(WHT);
  drawHeader(curUnsure);

  centerText(curLabel, 14, 2);
  if (curMaterial.length() > 0) {
    String m = curMaterial;
    if (m.length() > 21) m = m.substring(0, 21);  // one line = 21 characters
    centerText(m, 33, 1);
  }

  display.drawRect(0, 43, 98, 10, WHT);
  display.setTextSize(1);
  display.setCursor(102, 44);
  if (curScore >= 0) {
    display.fillRect(2, 45, (94 * curScore) / 100, 6, WHT);
    display.print(curScore);
    display.print("%");
  } else {
    display.print("--");
  }

  centerText("TIME " + withUnit(curMs, " ms"), 56, 1);
  display.display();
}

// STATS: totals, landfill %, bars (solid = confident, hollow = unsure)
void drawStatsPage() {
  display.clearDisplay();
  display.setTextColor(WHT);
  display.setTextSize(1);

  int unsureTotal = unsureCnt[0] + unsureCnt[1] + unsureCnt[2];
  display.setCursor(0, 0);
  display.print("SORTED ");
  display.print(total);
  rightText(String(unsureTotal) + " UNSURE", 0);

  int kept = confCnt[0] + unsureCnt[0] + confCnt[1] + unsureCnt[1];   // compost + recycle
  int saved = total ? (100 * kept) / total : 0;
  centerText("LANDFILL SAVED " + String(saved) + "%", 10, 1);

  int maxT = 1;
  for (int i = 0; i < 3; i++) if (confCnt[i] + unsureCnt[i] > maxT) maxT = confCnt[i] + unsureCnt[i];

  const int bottom = 54, maxH = 22, w = 28;
  for (int i = 0; i < 3; i++) {
    int x = 8 + i * 42;
    int tot = confCnt[i] + unsureCnt[i];
    int hTot  = (maxH * tot) / maxT;
    int hConf = (maxH * confCnt[i]) / maxT;
    if (unsureCnt[i] > 0 && hTot - hConf < 3) {            // keep the hollow part visible
      hTot = hConf + 3;
      if (hTot > maxH) { hTot = maxH; hConf = maxH - 3; }
    }
    if (hConf > 0) display.fillRect(x, bottom - hConf, w, hConf, WHT);
    if (hTot > hConf) display.drawRect(x, bottom - hTot, w, hTot - hConf, WHT);

    String c = String(tot);
    if (unsureCnt[i] > 0) c += "(" + String(unsureCnt[i]) + "?)";
    display.setCursor(x + w / 2 - (int)c.length() * 3, bottom - hTot - 9);
    display.print(c);

    display.setCursor(x + w / 2 - (int)strlen(SHORTS[i]) * 3, 57);
    display.print(SHORTS[i]);
  }
  display.drawFastHLine(0, bottom, 128, WHT);
  display.display();
}

// SYSTEM: inference times, confidence, uptime, offline badge
void drawSystemPage() {
  display.clearDisplay();
  display.setTextColor(WHT);
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.print("SYSTEM");
  display.fillRect(74, 0, 54, 9, WHT);
  display.setTextColor(BLK);
  display.setCursor(80, 1);
  display.print("OFFLINE");
  display.setTextColor(WHT);

  row("AVG INFER",  withUnit(msN ? (long)(sumMs / msN) : -1, " ms"), 14);
  row("LAST INFER", withUnit(curMs, " ms"), 24);
  row("FASTEST",    withUnit(fastestMs, " ms"), 34);
  row("AVG CONF",   withUnit(scoreN ? (long)(sumScore / scoreN) : -1, "%"), 44);

  unsigned long s = millis() / 1000;
  char up[12];
  snprintf(up, sizeof(up), "%02lu:%02lu", s / 60, s % 60);
  row("UPTIME", String(up), 54);
  display.display();
}

void drawPage() {
  if (page != 0) { inverted = false; display.invertDisplay(false); }
  if (page == 0)      drawResultPage();
  else if (page == 1) drawStatsPage();
  else                drawSystemPage();
}

// scanner used by IDLE and SCANNING
void drawScanner(int f, const char* msg, int speed) {
  display.clearDisplay();
  display.setTextColor(WHT);
  centerText("GARBAGE COLLECTORS", 0, 1);

  const int x0 = 34, y0 = 12, x1 = 94, y1 = 46, L = 8;
  display.drawFastHLine(x0, y0, L, WHT);         display.drawFastVLine(x0, y0, L, WHT);
  display.drawFastHLine(x1 - L + 1, y0, L, WHT); display.drawFastVLine(x1, y0, L, WHT);
  display.drawFastHLine(x0, y1, L, WHT);         display.drawFastVLine(x0, y1 - L + 1, L, WHT);
  display.drawFastHLine(x1 - L + 1, y1, L, WHT); display.drawFastVLine(x1, y1 - L + 1, L, WHT);

  const int span = y1 - y0 - 4;
  int p = (f * speed) % (2 * span);
  int sy = y0 + 2 + (p < span ? p : 2 * span - p);
  display.drawFastHLine(x0 + 3, sy, x1 - x0 - 5, WHT);

  display.setTextSize(1);
  display.setCursor((128 - ((int)strlen(msg) + 3) * 6) / 2, 54);
  display.print(msg);
  for (int i = 0; i < (f / 8) % 4; i++) display.print(".");
  display.display();
}

// NO CAMERA / MODEL ERROR from the material classifier app
void showStatus(const String& message) {
  mode = M_STATUS;
  modeAt = millis();
  inverted = false;
  display.invertDisplay(false);
  setLedsOff();
  display.clearDisplay();
  display.setTextColor(WHT);
  centerText("GARBAGE COLLECTORS", 0, 1);
  display.drawFastHLine(0, 10, 128, WHT);
  centerText(message, 28, message.length() <= 10 ? 2 : 1);
  centerText(message.indexOf("CAMERA") >= 0 ? "check the USB camera" : "see the app log", 54, 1);
  display.display();
  Monitor.println(String("STATUS ") + message);
}

void goIdle() {
  mode = M_IDLE;
  streak = 0;
  rapidStats = false;
  rapidAuto = false;
  itemOpen = false;
  modeAt = millis();
  idleSub = 0;
  frame = 0;
  inverted = false;
  display.invertDisplay(false);
  setLedsOff();
}
