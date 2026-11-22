// =====================================================================
//  animations.ino - 1 s per lane (T_ANIM_NORMAL / T_ANIM_RAPID). q = low score -> blinking "?"
// =====================================================================

// Each animation has ~25 frames, so frame time = animation length / 25 (1 s -> 40 ms).
// Every animation stops early when the next result arrives (inNewItem).
unsigned long animFrameMs() { return (rapidNow() ? T_ANIM_RAPID : T_ANIM_NORMAL) / 25; }

void drawQuestion(int f, bool q) {
  if (q && (f / 4) % 2 == 0) {
    display.setTextSize(3);
    display.setCursor(104, 10);
    display.print("?");
  }
}

// three arrows chasing round a circle
void animRecycle(bool q) {
  const int cx = 64, cy = 23, r = 17;
  for (int f = 0; f < 24; f++) {
    display.clearDisplay();
    display.drawCircle(cx, cy, r, WHT);
    float base = f * 0.26f;
    for (int k = 0; k < 3; k++) {
      float a = base + k * 2.094f;
      float px = cx + r * cosf(a), py = cy + r * sinf(a);
      float tx = -sinf(a), ty = cosf(a), nx = cosf(a), ny = sinf(a);
      display.fillTriangle((int)(px + tx * 7),          (int)(py + ty * 7),
                           (int)(px - tx * 2 + nx * 5), (int)(py - ty * 2 + ny * 5),
                           (int)(px - tx * 2 - nx * 5), (int)(py - ty * 2 - ny * 5), WHT);
    }
    drawQuestion(f, q);
    centerText("RECYCLE", 48, 2);
    display.display();
    pause(animFrameMs());
    if (inNewItem) return;                   // next result is here: skip the rest
  }
}

// sprout grows out of the soil
void animCompost(bool q) {
  const int gx = 64, gy = 42;
  for (int f = 0; f < 26; f++) {
    display.clearDisplay();
    display.fillRoundRect(34, gy, 60, 5, 2, WHT);
    int h = f < 14 ? f * 2 : 26;
    display.drawFastVLine(gx, gy - h, h, WHT);
    display.drawFastVLine(gx + 1, gy - h, h, WHT);
    if (f >= 12) {
      int lr = clampi(f - 11, 0, 7);
      display.fillCircle(gx - lr - 1, gy - 16, lr, WHT);
      display.fillCircle(gx + lr + 2, gy - 23, lr, WHT);
    }
    drawQuestion(f, q);
    centerText("COMPOST", 48, 2);
    display.display();
    pause(animFrameMs());
    if (inNewItem) return;                   // next result is here: skip the rest
  }
}

void drawBin(int x, int y, float lid) {
  display.drawRect(x, y, 28, 22, WHT);
  display.drawFastVLine(x + 9,  y + 4, 14, WHT);
  display.drawFastVLine(x + 18, y + 4, 14, WHT);
  int hx = x - 2, hy = y - 2;
  int ex = hx + (int)(32 * cosf(lid)), ey = hy + (int)(32 * sinf(lid));
  display.drawLine(hx, hy, ex, ey, WHT);
  display.drawLine(hx, hy - 1, ex, ey - 1, WHT);
}

// bin lid opens, item drops in, lid closes
void animTrash(bool q) {
  const int bx = 50, by = 22;
  for (int f = 0; f < 24; f++) {
    display.clearDisplay();
    float lid;
    if (f < 6)       lid = -0.2f * f;
    else if (f < 16) lid = -1.2f;
    else             lid = -1.2f + 0.15f * (f - 15);
    if (lid > 0) lid = 0;
    drawBin(bx, by, lid);
    if (f >= 6 && f < 16) display.fillRect(bx + 11, (f - 6) * 3, 6, 6, WHT);
    drawQuestion(f, q);
    centerText("TRASH", 48, 2);
    display.display();
    pause(animFrameMs());
    if (inNewItem) return;                   // next result is here: skip the rest
  }
}

void playAnimation() {
  display.invertDisplay(false);
  if (curLane == 0)      animCompost(curUnsure);
  else if (curLane == 1) animRecycle(curUnsure);
  else                   animTrash(curUnsure);
}
