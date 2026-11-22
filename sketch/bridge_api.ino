// =====================================================================
//  bridge_api.ino - the functions Python calls through the Bridge
//  They only store values; loop() calls commitIncoming() to do the work,
//  so drawing and animations never run inside a Bridge call.
// =====================================================================

int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

int laneIndex(const String& up) {
  for (int i = 0; i < 3; i++) if (up == LANES[i]) return i;
  return -1;
}

// Used only when Python sends a material instead of a lane.
String materialToLabel(String m) {
  m.toLowerCase();
  if (m == "cardboard" || m == "paper" ||
      m == "metal" || m == "glass")                                  return "RECYCLE";   // plastic -> TRASH (team rule)
  if (m == "food" || m == "food_organic" || m == "organic" || m == "biological" ||
      m == "greasy_paper" || m == "compost")                         return "COMPOST";
  return "TRASH";                                                    // unknown -> safe default
}

// ---------------- display ----------------
// lane = COMPOST / RECYCLE / TRASH (decided in Python), object = what it is ("bottle")
void display_item(String lane, String object, int score, int ms) {
  inStream = false;
  inText = lane;
  inObject = object;
  inScore = score < 0 ? -1 : clampi(score, 0, 100);
  inMs = ms;
  inNewItem = true;
  inChangedAt = millis();
  inDirty = true;
}

// text = a lane or a material ("cardboard" -> RECYCLE)
void display_label(String text) { display_item(text, "", -1, -1); }
void display_score(int s)  { inScore = s < 0 ? -1 : clampi(s, 0, 100); inChangedAt = millis(); inDirty = true; }
void display_time(int ms)  { inMs = ms; inChangedAt = millis(); inDirty = true; }
void display_result(String text, int s, int ms) { display_item(text, "", s, ms); }
// Material classifier app: about one call per second, see sketch.ino
void show_result(String lane, String material, int score, int ms) {
  lane.trim();
  lane.toUpperCase();
  if (laneIndex(lane) < 0) lane = materialToLabel(material);   // unknown or empty lane
  inText = lane;
  inObject = material;
  inScore = score < 0 ? -1 : clampi(score, 0, 100);
  inMs = ms;
  inStream = true;
  inNewItem = true;
  inChangedAt = millis();
  inDirty = true;
}
void show_status(String message) { inStatusText = message; inStatus = true; }

void display_scanning()    { inScan = true; inItemStart = true; }
void reset_stats()         { inReset = true; }
void set_rapid_fire(int on) { rapidForced = (on != 0); }

bool rapidNow() { return rapidForced || rapidAuto; }

// ---------------- belt ----------------
void belt_on()             { beltOn = true;  beltStopAt = 0; motorApply(); }
void belt_stop()           { beltOn = false; beltStopAt = 0; motorApply(); }
void belt_speed(int s)     { beltSpeed = clampi(s, 0, 255); motorApply(); }
void belt_run_for(int ms) {
  beltOn = true;
  beltStopAt = millis() + (unsigned long)ms;
  if (beltStopAt == 0) beltStopAt = 1;
  motorApply();
}
// After a decision: carry the item off. One belt for now, so every lane does the same;
// a sorting arm or per-lane timing would go here.
void run_conveyor(String lane) {
  Monitor.println(String("run_conveyor ") + lane);
  belt_run_for((int)T_CONVEYOR);
}

// provide_safe: every call runs in the main loop (between loop() passes), not in the
// Bridge's own thread, whose stack is only 500 bytes. Safer for the String work below.
void provideBridgeFunctions() {
  Bridge.provide_safe("show_result", show_result);
  Bridge.provide_safe("show_status", show_status);
  Bridge.provide_safe("display_item", display_item);
  Bridge.provide_safe("display_result", display_result);
  Bridge.provide_safe("display_label", display_label);
  Bridge.provide_safe("display_score", display_score);
  Bridge.provide_safe("display_time", display_time);
  Bridge.provide_safe("display_scanning", display_scanning);
  Bridge.provide_safe("reset_stats", reset_stats);
  Bridge.provide_safe("set_rapid_fire", set_rapid_fire);
  Bridge.provide_safe("run_conveyor", run_conveyor);
  Bridge.provide_safe("belt_on", belt_on);
  Bridge.provide_safe("belt_stop", belt_stop);
  Bridge.provide_safe("belt_speed", belt_speed);
  Bridge.provide_safe("belt_run_for", belt_run_for);
}

// ---------------- turning calls into a result (runs in loop) ----------------
void addMetrics() {
  if (!curMsCounted && curMs >= 0) {
    sumMs += curMs; msN++;
    if (fastestMs < 0 || curMs < fastestMs) fastestMs = curMs;
    curMsCounted = true;
  }
  if (!curScoreCounted && curScore >= 0) {
    sumScore += curScore; scoreN++;
    curScoreCounted = true;
  }
}

// the item on screen turned out to be another lane: move its count
void relabelCurrent(int li, const String& object, int sc, int ms) {
  if (curUnsure) unsureCnt[curLane]--; else confCnt[curLane]--;
  curLane = li;
  curLabel = LANES[li];
  if (curUnsure) unsureCnt[li]++; else confCnt[li]++;
  updateCurrent(object, sc, ms);
}

// score / time / material changed for the item already on screen
void updateCurrent(const String& object, int sc, int ms) {
  if (object.length() > 0) curMaterial = object;
  bool was = curUnsure;
  curScore = sc;
  curMs = ms;
  curUnsure = (sc >= 0 && sc < CONF_MIN);
  if (was != curUnsure) {
    if (was) { unsureCnt[curLane]--; confCnt[curLane]++; }
    else     { confCnt[curLane]--;   unsureCnt[curLane]++; }
  }
  addMetrics();
  setLedsForResult();
  if (mode == M_RESULT && page == 0) drawResultPage();
}

void commitIncoming() {
  bool newItem = inNewItem, stream = inStream;
  bool itemStart = inItemStart;
  if (newItem && stream) inItemStart = false;
  inNewItem = false;
  inDirty = false;
  String text = inText, object = inObject;
  int sc = inScore, ms = inMs;
  unsigned long now = millis();

  if (newItem && stream && itemStart) {
    // display_scanning came first: this is a new item, no confirmation needed
    pendingLane = -1;
    pendingCount = 0;
    itemOpen = true;
  } else if (newItem && stream) {
    // show_result repeats every second while an item is in view
    String up = text;
    up.trim();
    up.toUpperCase();
    int li = laneIndex(up);
    bool showing = (mode == M_RESULT) && curLane >= 0 && now - lastSeenAt < T_SAME_ITEM;
    if (showing && (li == curLane || itemOpen)) {    // same item: refresh, don't count
      lastSeenAt = now;
      pendingLane = -1;
      object.trim();
      curMsCounted = curScoreCounted = false;        // count this reading's time and score
      if (li == curLane)     updateCurrent(object, sc, ms);
      else if (sc > curScore) relabelCurrent(li, object, sc, ms);   // keep the surest reading
      return;
    }
    itemOpen = false;
    if (li != pendingLane || now - pendingAt > T_SAME_ITEM) pendingCount = 0;
    pendingLane = li;
    pendingAt = now;
    if (++pendingCount < STREAM_CONFIRM) return;     // new lane: wait for a repeat
    pendingLane = -1;                                // confirmed: new item
    pendingCount = 0;
  }

  if (newItem) {
    lastSeenAt = now;
    // rapid fire: on when this result came soon after the last one, off after a long gap
    unsigned long gap = millis() - lastItemAt;
    if (AUTO_RAPID && total > 0 && gap < T_RAPID_TRIGGER) rapidAuto = true;
    else if (gap >= T_RAPID_PAUSE)                        rapidAuto = false;
    streak = rapidNow() ? streak + 1 : 1;

    text.trim();
    object.trim();
    String up = text;
    up.toUpperCase();
    int li = laneIndex(up);
    if (li >= 0) { curLabel = LANES[li]; curMaterial = object; }
    else         { curMaterial = text; curLabel = materialToLabel(text); li = laneIndex(curLabel); }

    curLane = li;
    curScore = sc;
    curMs = ms;
    curUnsure = (sc >= 0 && sc < CONF_MIN);
    curMsCounted = curScoreCounted = false;

    total++;
    if (curUnsure) unsureCnt[li]++; else confCnt[li]++;
    addMetrics();

    Monitor.println(curLabel + " | " + curMaterial + " | " + String(curScore) + "% | " +
                    String(curMs) + " ms" + (curUnsure ? " | HUMAN CHECK" : ""));

    rapidStats = false;
    setLedsForResult();
    mode = M_RESULT;                 // so the white LED goes off during the animation
    playAnimation();

    page = 0;
    pageAt = lastItemAt = flashAt = millis();
    inverted = false;
    drawResultPage();
  } else if (curLane >= 0) {
    // score or time arrived after the label: update the current item
    updateCurrent("", sc, ms);
  }
}

// ---------------- DEMO_MODE 1: fake items, no Python needed ----------------
// 3 slow items (SCANNING first, 12 s apart), then a burst of 8 items 3 s apart
// (back-to-back items on our belt), which switches rapid fire on by itself. Repeats.
void runDemo(unsigned long now) {
  static unsigned long at = 0;
  static int n = 0;
  static bool scanned = false;
  static const char* what[7]  = {"cardboard", "food", "metal", "wrapper", "plastic", "paper", "glass"};
  static const int   score[7] = {88, 91, 76, 83, 42, 55, 90};
  static const int   ms[7]    = {34, 38, 41, 29, 36, 33, 31};
  int step = n % 11;
  unsigned long gap = (step < 3) ? 12000UL : (step == 3 ? 12000UL : 3000UL);
  if (gap > T_RAPID_TRIGGER && !scanned && now - at > gap - 1500) {      // item seen before its result
    display_scanning();
    scanned = true;
  }
  if (now - at > gap) {
    int i = n % 7;
    display_result(what[i], score[i], ms[i]);
    belt_run_for(1500);                                     // does nothing with MOTOR_MODE 0
    n++; at = now; scanned = false;
  }
}
