#include <Arduino.h>

// Motor
#define IN1 26
#define IN2 27
#define ENA 25
#define MOTOR_CHANNEL 0
#define MOTOR_FREQ 5000
#define MOTOR_RES 8
#define PWM_MIN 150
#define PWM_MAX 255

// Servo (manual PWM, no library, separate channel/timer from motor)
#define SERVO_PIN 19
#define SERVO_CHANNEL 4
#define SERVO_FREQ 50
#define SERVO_RES 16
#define SERVO_MIN_US 500
#define SERVO_MAX_US 2500

// Encoder
#define ENC_A 34
#define ENC_B 35

volatile long encoderTicks = 0;
void IRAM_ATTR encoderISR() {
  int a = digitalRead(ENC_A);
  int b = digitalRead(ENC_B);
  encoderTicks += (a == b) ? 1 : -1;
}

String inputBuffer = "";
unsigned long lastFeedback = 0;
const unsigned long FEEDBACK_INTERVAL_MS = 50;

void motorStop() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  ledcWrite(MOTOR_CHANNEL, 0);
}

void setMotor(int throttle) {
  throttle = constrain(throttle, -255, 255);
  if (throttle == 0) { motorStop(); return; }
  int mag = abs(throttle);
  int pwm = constrain(map(mag, 1, 255, PWM_MIN, PWM_MAX), PWM_MIN, PWM_MAX);
  if (throttle > 0) { digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW); }
  else              { digitalWrite(IN1, LOW);  digitalWrite(IN2, HIGH); }
  ledcWrite(MOTOR_CHANNEL, pwm);
}

void setServoAngle(int angle) {
  angle = constrain(angle, 0, 360);
  int pulse_us = map(angle, 0, 360, SERVO_MIN_US, SERVO_MAX_US);
  uint32_t maxDuty = (1UL << SERVO_RES) - 1;
  uint32_t duty = (uint32_t)(((float)pulse_us / 20000.0f) * maxDuty);
  ledcWrite(SERVO_CHANNEL, duty);
}

void processCommand(String cmd) {
  cmd.trim();
  if (cmd.length() < 3) return;
  char type = cmd.charAt(0);
  String value = cmd.substring(2);

  if (type == 'M') {
    int throttle = value.toInt();
    setMotor(throttle);
    Serial.print("AM,"); Serial.println(throttle);
  } else if (type == 'S') {
    int angle = value.toInt();
    setServoAngle(angle);
    Serial.print("AS,"); Serial.println(angle);
  }
}

void setup() {
  Serial.begin(115200);

  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  ledcSetup(MOTOR_CHANNEL, MOTOR_FREQ, MOTOR_RES);
  ledcAttachPin(ENA, MOTOR_CHANNEL);

  ledcSetup(SERVO_CHANNEL, SERVO_FREQ, SERVO_RES);
  ledcAttachPin(SERVO_PIN, SERVO_CHANNEL);
  setServoAngle(90);

  pinMode(ENC_A, INPUT);
  pinMode(ENC_B, INPUT);
  attachInterrupt(digitalPinToInterrupt(ENC_A), encoderISR, CHANGE);

  motorStop();
  Serial.println("READY");
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') { processCommand(inputBuffer); inputBuffer = ""; }
    else inputBuffer += c;
  }

  unsigned long now = millis();
  if (now - lastFeedback >= FEEDBACK_INTERVAL_MS) {
    lastFeedback = now;
    Serial.print("ENC,");
    Serial.println(encoderTicks);
  }
}
