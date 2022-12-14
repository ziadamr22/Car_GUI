#include <Servo.h>
#define RED_PIN 11  //used common cathod RGB if u used common anode then change TurnON_RGB to 0 and TurnON_RGB to 255
#define GREEN_PIN 10
#define BLUE_PIN 9
#define C_Motor1_dir_A1 2 
#define C_Motor1_dir_A2 4
#define C_Motor2_dir_A1 7
#define C_Motor2_dir_A2 8
#define C_Motor1_Speed_A3 3  //PWM pin
#define C_Motor2_Speed_A3 6  //PWM pin
#define ESC_PIN_Motor1 3  //pwm pin
#define ESC_PIN_Motor2 6  //pwm pin
#define pot A0
#define joyX A2
#define joyY A3
#define VoltageSensor A4
#define CurrentSensor A5
#define TurnON_RGB HIGH
#define TurnOFF_RGB LOW
#define Low_Speed 85
#define Med_Speed 170
#define High_Speed 255
#define Mp_resolution 1023.0
#define factor 0.2      
#define refVoltage 5.0  
#define MinVoltage 15
//#define MaxVoltage           24
#define MinPulseWidth 1000
#define MaxPulseWidth 2000
#define Sensitivity 0.185
#define NoLoadVolt 2.5
#define MaxCurrent 4

int MotorSpeed;
int X_value, Y_value;
int pwmOutput;
int Voltage, Current;
int Speed;
int ESC_Speed;
char dir;
char motor;
String key;
unsigned int t_start;
unsigned int t_end;
const byte High_Consumed_I_LED = 13;
const byte Low_InVoltage_LED = 5;

Servo ESC_Motor1;
Servo ESC_Motor2;

void setup() {
  Serial.begin(9600);
  // BT.begin(9600);
  t_start = millis();
  // initializing indicator pins
  pinMode(RED_PIN, OUTPUT);
  pinMode(GREEN_PIN, OUTPUT);
  pinMode(BLUE_PIN, OUTPUT);
  // initializing Motor direction pins
  pinMode(C_Motor1_dir_A1, OUTPUT);
  pinMode(C_Motor1_dir_A2, OUTPUT);
  pinMode(C_Motor2_dir_A1, OUTPUT);
  pinMode(C_Motor2_dir_A2, OUTPUT);
  // initializing Motor speed pins
  pinMode(C_Motor1_Speed_A3, OUTPUT);
  pinMode(C_Motor2_Speed_A3, OUTPUT);
  // initializing input voltage and consumed current indicator pins
  pinMode(Low_InVoltage_LED, OUTPUT);
  pinMode(High_Consumed_I_LED, OUTPUT);
  //initializing ESC motor driver
  ESC_Motor1.attach(ESC_PIN_Motor1, MinPulseWidth, MaxPulseWidth);  // the pulse will be between (1ms to 2ms)
  ESC_Motor2.attach(ESC_PIN_Motor2, MinPulseWidth, MaxPulseWidth);
  ESC_Motor1.write(0);  //send stop signal to ESC to arm it then delay to allow ESC recognize this signal
  ESC_Motor2.write(0);
  delay(5000);
}
void loop() {

  Serial.println("Hello")  ;
  while(!Serial.available());
  key = Serial.readString();
  if (key[0]=='s'){
    // String temp = String(key[1]);
    Speed = String(key[1]).toInt();
    SetMotor_Speed(Speed);
    RGB_SpeedIndicator(Speed);
  }
  else if (key[0]=='m'){
    motor = key[1];
  }
  else if (key[0]=='d'){
    dir = key[1];
  }
  if(motor =='1')
  {
            if (dir == '3')  //Signal sent from python to move forward = 3
          {
            Forward();
          } else if (dir == '4' )  //Signal sent from python to move backward = 4
          {
            Backward();
          } else if (dir == '5')  //Signal sent from python to move right = 5
          {
            Right();
           } else if (dir == '6')  //Signal sent from python to move left = 6
          {
            Left();
          } else if (dir == '7')  //Signal sent from python to  stop = 7
          {
            Stop();
          }
     }
    else if (motor == "2") {
      ESC_Speed = analogRead(pot);
      Speed = SetESC_Speed(ESC_Speed);  
      Speed = map(Speed, 0, 180, 0, 2);
      RGB_SpeedIndicator(Speed);
    }
    // Reading voltage and current sensor values
    Voltage = analogRead(VoltageSensor);
    Voltage_Sensor(Voltage);
    Current = analogRead(CurrentSensor);
    Current_Sensor(Current);
    Serial.println();
  
}
void RGB_SpeedIndicator(int MotorSpeed) {
  if (MotorSpeed == 0) {
    digitalWrite(GREEN_PIN, TurnON_RGB);
    digitalWrite(RED_PIN, TurnOFF_RGB);
    digitalWrite(BLUE_PIN, TurnOFF_RGB);

  } else if (MotorSpeed == 1) {
    digitalWrite(RED_PIN, TurnOFF_RGB);
    digitalWrite(GREEN_PIN, TurnOFF_RGB);
    digitalWrite(BLUE_PIN, TurnON_RGB);

  }

  else if (MotorSpeed == 2) {
    digitalWrite(RED_PIN, TurnON_RGB);
    digitalWrite(GREEN_PIN, TurnOFF_RGB);
    digitalWrite(BLUE_PIN, TurnOFF_RGB);

  }
}


void SetMotor_Speed(int Speed) {
  
  if (Speed == 0) {
    analogWrite(C_Motor1_Speed_A3, Low_Speed);
    analogWrite(C_Motor2_Speed_A3, Low_Speed);
  } else if (Speed == 1) {
    analogWrite(C_Motor1_Speed_A3, Med_Speed);
    analogWrite(C_Motor2_Speed_A3, Med_Speed);

  }

  else if (Speed == 2) {
    analogWrite(C_Motor1_Speed_A3, High_Speed);
    analogWrite(C_Motor2_Speed_A3, High_Speed);

  }
}

void Forward() {
  digitalWrite(C_Motor1_dir_A1, HIGH);  //  Motor1 moves forward
  digitalWrite(C_Motor1_dir_A2, LOW);
  digitalWrite(C_Motor2_dir_A1, HIGH);  //  Motor2 moves forward
  digitalWrite(C_Motor2_dir_A2, LOW);
  delay(20);
}

void Backward() {
  digitalWrite(C_Motor1_dir_A1, LOW);  //  Motor1 moves backward
  digitalWrite(C_Motor1_dir_A2, HIGH);
  digitalWrite(C_Motor2_dir_A1, LOW);  //  Motor2 moves backward
  digitalWrite(C_Motor2_dir_A2, HIGH);
  delay(20);
}

void Right() {
  digitalWrite(C_Motor1_dir_A1, LOW);  //  Motor1 moves backward
  digitalWrite(C_Motor1_dir_A2, HIGH);
  digitalWrite(C_Motor2_dir_A1, HIGH);  //  Motor2 moves forward
  digitalWrite(C_Motor2_dir_A2, LOW);
  delay(20);
}

void Left() {
  digitalWrite(C_Motor1_dir_A1, HIGH);  //  Motor1 moves forward
  digitalWrite(C_Motor1_dir_A2, LOW);
  digitalWrite(C_Motor2_dir_A1, LOW);  //  Motor2 moves backward
  digitalWrite(C_Motor2_dir_A2, HIGH);
  delay(20);
}

void Stop() {
  digitalWrite(C_Motor1_dir_A1, LOW);  // to stop motion
  digitalWrite(C_Motor1_dir_A2, LOW);
  digitalWrite(C_Motor2_dir_A1, LOW);
  digitalWrite(C_Motor2_dir_A2, LOW);
  delay(20);
}

void Voltage_Sensor(int Voltage) {
  float mapped_voltage = 0.0;
  float InVoltage = 0.0;
  mapped_voltage = MapFunc(Voltage,0.0,Mp_resolution,0.0,refVoltage);
  InVoltage = mapped_voltage / factor;
  if (InVoltage < MinVoltage)  // if the voltage became below min voltage then indicator led will blink until input voltage becomes above min voltage
  {
   Blink_LED(Low_InVoltage_LED,250); 
  }
  Serial.print'v');
  Serial.print(InVoltage);
}

int SetESC_Speed(int Pot_Value) {
  int Speed;
  Speed = map(Pot_Value, 0, 1023, 0, 180);  // varying the potentiometer means increasing and decreasing the speed
  ESC_Motor1.write(Speed);
  ESC_Motor2.write(Speed);
  return Speed;
}

void Current_Sensor(int Current) {
  float voltage = 0.0;
  float consumed_current = 0.0;
  voltage = MapFunc(Current,0.0,Mp_resolution,0.0,refVoltage);
  consumed_current = ((voltage - NoLoadVolt) / Sensitivity);
  if (consumed_current > MaxCurrent)  // if consumed current is above max current then a blinking LED will indicate this
  {
    Blink_LED(High_Consumed_I_LED,250);
  }
  Serial.print('c');
  Serial.print(consumed_current);
}

float MapFunc(float sensor ,float InputMin, float InputMax , float OutputMin,float OutputMax )
{ 
  float slope = 0.0 ;
  float value = 0.0 ;
  float Input = 0.0 ;         
  float Output = 0.0;         
  InputMin = 0;         
  OutputMin = 0;        
  /* y = mx + c */      
  /*slope = (y2 - y1) / (x2 - x1)*/
  slope = (OutputMax - OutputMin)/(InputMax - InputMin);
  Input = InputMax - InputMin ;
  Output = (slope*Input);
  InputMax = Input ;
  OutputMax = Output;
  value = (OutputMax * sensor )/InputMax;
  
  return value;
}

void Blink_LED(byte led,unsigned long period)
{
  t_end = millis();
  if(t_end - t_start >= period){
    digitalWrite(led,!digitalRead(led));
    t_start = t_end;
  }
}
