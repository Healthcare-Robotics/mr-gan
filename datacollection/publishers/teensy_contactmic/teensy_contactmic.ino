int heartbeatPin = 13;

void setup() {
    Serial.begin(115200);
    analogReference(DEFAULT);
    analogReadResolution(12);
    analogWriteResolution(12);
    pinMode(heartbeatPin, OUTPUT);
    digitalWrite(heartbeatPin, HIGH);
}

void loop() {
    Serial.println(analogRead(A0));
    Serial.send_now();
}
