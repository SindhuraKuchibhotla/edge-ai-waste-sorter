// =====================================================================
//  belt.ino - motor control. MOTOR_MODE 0 = the motor runs on its own
//  resistors and the code can't switch it; the belt functions just log.
// =====================================================================

void motorApply() {
#if MOTOR_MODE == 1
  analogWrite(MOTOR_A, beltOn ? beltSpeed : 0);
#elif MOTOR_MODE == 2
  analogWrite(MOTOR_A, beltOn ? beltSpeed : 0);
  digitalWrite(MOTOR_B, LOW);
#endif
}

// stops a belt_run_for() when its time is up
void serviceBelt() {
  if (beltStopAt && (long)(millis() - beltStopAt) >= 0) {
    beltOn = false;
    beltStopAt = 0;
    motorApply();
  }
}
