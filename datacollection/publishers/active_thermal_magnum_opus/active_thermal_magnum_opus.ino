#include <math.h>

IntervalTimer buck_control_timer;
IntervalTimer send_data_timer;
IntervalTimer temperature_control_timer;

const int heartBeatPin = 13;

const int outpin_source1 = 3;     /// change me
const int input_pin_supply1 = 22; /// change me

const int active_thermistor_pin1 = 14;
const int active_thermistor_pin2 = 15;

#define minV 7
#define maxV 13

int incomingValue;
volatile float Setpoint1mv = 10000;
volatile float Tset = 55.0; //C
const float mv_coeff = 4.922;

int serialVals[2];
int serialNdx = 0;


const float kp = 0.5; // 0-4095 DC units per mv of error
const float ki = 1; // 0-4095 DC units per mv of error
const float kd = 0; // 0-4095 DC units per mv of error

volatile float Te = 0;
volatile float Te_last = 0;
volatile float Te_sum = minV * 1000.0;
const float temperature_dt = 0.1;

const float temperature_kp = 1500.0; // mv/(K)
float temperature_ki = 300.0; // mv/(K s)
float temperature_kd = 30.0; // mv/(K/s)
const float TI_max = maxV * 1000.0;
volatile int control = 1;
volatile int LED = 1;

const float len = 9.0;
volatile int Input1mv[]= {0,0,0,0,0,0,0,0,0,0};
volatile int i = 0;
float Inmv = 19960;

volatile float Input1mv_avg = 0;
volatile float Delta1mv;
float I_max = 1500;
volatile float Delta1mv_sum = 1000;
volatile float Delta1mv_last = 0;
volatile float pwm1;

float buck_dt = 0.001;
volatile float out1 = 0;

float data_dt = 0.01;

// Convert voltage to temperature in celcius
float temperature(int a, float Vsupp, float Rref) {
    const float Vref = 3.3;
    float Vin = ((float)a)/4095.0*Vref;
    const float T1 = 288.15;
    const float Beta = 3406;
    const float R1 = 14827;
    Vin = constrain(Vin,0.001, 3.3);
    float RT = constrain(Rref*((Vsupp/Vin) - 1),1000, 20000);
    float TC = (T1*Beta/log(R1/RT))/(Beta/log(R1/RT) - T1) - 273.15;
    return TC;
}

// Return temperature of active temperature sensor in celcius
float active(void) {
    return temperature(analogRead(active_thermistor_pin1), Input1mv_avg*0.001, 1000);
}

void temperature_controller(void) {
    float act = active();

    if (act > 20 && control) {
        Te = Tset - act;
        if (abs(Te) < 0.5) {
            LED = !LED;
            digitalWrite(heartBeatPin,LED);
        }

        Te_sum = constrain(Te_sum + Te*temperature_ki*temperature_dt , minV*1000.0, maxV*1000.0 - temperature_kp*Te);
        Setpoint1mv = constrain(temperature_kp*Te + Te_sum + temperature_kd*(Te - Te_last)/temperature_dt, minV*1000.0, maxV*1000.0);
        Te_last = Te;
    } else {
        digitalWrite(heartBeatPin,HIGH);
    }
}

void buck_controller(void) {
    for (i = 0; i < len; i++) {
        Input1mv[i] = Input1mv[i+1];
    }

    Input1mv[(int)len] = int(analogRead(input_pin_supply1)*mv_coeff);
    Input1mv_avg = 0;
    for (i = 0; i <= len; i++) {
        Input1mv_avg += Input1mv[i]/(len + 1);
    }
    Delta1mv = Setpoint1mv - Input1mv_avg ;
    Delta1mv_sum = constrain(Delta1mv_sum + Delta1mv*buck_dt*ki, -I_max, I_max);
    pwm1 = constrain(Delta1mv*kp + Delta1mv_sum + (Delta1mv - Delta1mv_last)*kd,0,4095);
    Delta1mv_last = Delta1mv;
    analogWrite(outpin_source1, 4095 - pwm1);
}

void send_data(void) {
    int temp1 = analogRead(active_thermistor_pin1);
    // int temp2 = analogRead(active_thermistor_pin2);
    Serial.print(temp1); Serial.print(",");
    Serial.println(temperature(temp1, Input1mv_avg*0.001, 1000)); // Serial.print(",");
    // Serial.print(active()); Serial.print(",");
    // Serial.print(Setpoint1mv*0.001); Serial.print(",");
    // Serial.println(Input1mv_avg*0.001);
}

void setup() {
    Serial.begin(115200);
    Serial.setTimeout(10);
    analogReference(DEFAULT);
    analogReadResolution(12);
    analogWriteResolution(12);

    pinMode(heartBeatPin,OUTPUT);
    digitalWrite(heartBeatPin,HIGH);

    analogWriteFrequency(outpin_source1, 31250);


    pwm1 = 0;
    analogWrite(outpin_source1, pwm1);
    buck_control_timer.priority(0);
    buck_control_timer.begin(buck_controller, buck_dt*1E6);  // Run controller every X microseconds

    send_data_timer.priority(1);
    send_data_timer.begin(send_data, data_dt*1E6);  // Run controller every X microseconds

    temperature_control_timer.priority(2);
    temperature_control_timer.begin(temperature_controller, temperature_dt*1E6);  // Run controller every X microseconds
}

void loop() {
    delay(100000);
}

void serialEvent() {
    char command = (char) Serial.read();

    if (command == 'C')
        control = 1;
    if (command == 'H')
        control = 0;

    if (command == 'V') {
        int parsedVal = Serial.parseInt();
        if (parsedVal > 1000) {
            Setpoint1mv = constrain(parsedVal, 1000, 14000);
            control = 0;
        }
    }

    if (command == 'T') {
        int parsedVal = Serial.parseInt();
        if (parsedVal > 25000) {
            Tset = constrain((float) parsedVal * 0.001, 0, 55);
            control = 1;
        }
    }

    if (command == 'K') {
        int parsedVal = Serial.parseInt();
        if (parsedVal >= 1)
            temperature_ki = constrain(parsedVal, 1, 400);
    }
}

