// #include <Keypad.h>
// #include <SoftwareSerial.h>
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
#define Low_InVoltage_LED 5
//#define HIGH_InVoltage_LED   12
#define pot A0
//#define C_Motor_pot          A1
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
#define factor 0.2      // this voltage sensor consists of a voltage divider circuit and its factor =R2/(R1+R2) = 7.5k/(30k+7.5k) = 0.2 where R2 is connected in parallel with Vout
#define refVoltage 5.0  // arduino voltage
#define MinVoltage 15
//#define MaxVoltage           24
#define MinPulseWidth 1000
#define MaxPulseWidth 2000
#define Sensitivity 0.185
#define NoLoadVolt 2.5
#define High_Consumed_I_LED 13
#define MaxCurrent 4
// SoftwareSerial  BT(2,3);
int MotorSpeed;
int X_value, Y_value;
int pwmOutput;
int Voltage, Current;
int Speed;
int ESC_Speed;
char dir;
char motor;
String key;
const byte ROWS = 4;  //four rows
const byte COLS = 4;  //four columns
// char keys[ROWS][COLS] = {
//   {'1','2','3','A'},
//   {'4','5','6','B'},
//   {'7','8','9','C'},
//   {'*','0','#','D'}
// };
byte rowPins[ROWS] = { 0, 1, 5, 6 };    //connect to the row pinouts of the keypad
byte colPins[COLS] = { 7, 8, 12, 13 };  //connect to the column pinouts of the keypad
// Keypad keypad = Keypad( makeKeymap(keys), rowPins, colPins, ROWS, COLS );  // struct to pass (keys , rowpins,rows and cols) values to it
Servo ESC_Motor1;
Servo ESC_Motor2;
void setup() {
  Serial.begin(9600);
  // BT.begin(9600);
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
  // X_value = analogRead(joyX);
  // Y_value = analogRead(joyY);
  //pwmOutput = analogRead(pot);
  //pwmOutput = map(pwmOutput, 0, 1023, 0 , 255);
  //analogWrite(C_Motor1_Speed_A3, pwmOutput); // Send PWM signal to L298N Enable pin
  //  key = keypad.getKey();
  Serial.println("Bluetooth Ready!");
  while(!Serial.available());
  key = Serial.readString();
  Serial.println(key);
  // Voltage = analogRead(VoltageSensor);
  // float mapped_voltage = 0.0;
  // float InVoltage = 0.0;
  // mapped_voltage = ((Voltage * refVoltage) / Mp_resolution);
  // InVoltage = mapped_voltage / factor;
  // Serial.println(InVoltage); 
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
  Serial.println(motor);
  Serial.println(dir);
  if(motor =='1')
  {
            if (dir == '3')  //Signal sent from python to move forward = 
          {
            Forward();
            Serial.println("forward");
          } else if (dir == '4' )  //Signal sent from python to move backward = 4
          {
            Backward();
            Serial.println("back");
          } else if (dir == '5')  //Signal sent from python to move right = 5
          {
            Right();
            Serial.println("Right");
          } else if (dir == '6')  //Signal sent from python to move left = 6
          {
            Left();
            Serial.println("left");
          } else if (dir == '7')  //Signal sent from python to  stop = 7
          {
            Stop();
            Serial.println("stop");
          }
     }
    else if (motor == "2") {
      ESC_Speed = analogRead(pot);
      Speed = SetESC_Speed(ESC_Speed);  // will be modified
      Speed = map(Speed, 0, 180, 0, 2);
      RGB_SpeedIndicator(Speed);
    }


    //analogWrite(C_Motor_Speed_A3, pwmOutput); // Send PWM signal to L298N Enable pin
    //  key = keypad.getKey();
    //  if (key)
    //  {
    //   SetMotor_Dir(key);
    //  }
    // MotorSpeed = analogRead(pot);
    //MotorSpeed = key;
    //Speed = SetMotor_Speed(MotorSpeed);
    ESC_Speed = analogRead(pot);
    Speed = SetESC_Speed(ESC_Speed);
   // RGB_SpeedIndicator(key);
    //Current = analogRead(CurrentSensor);
  
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

//void SetMotor_Dir(char key)
//{
//X_value = map(X_value,0,1023,0,3);                  here,will replace joystick with keypad and below condition based on keypad
//Y_value = map(Y_value,0,1023,0,3);
//}

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
  mapped_voltage = ((Voltage * refVoltage) / Mp_resolution);
  InVoltage = mapped_voltage / factor;
  Serial.println(InVoltage);   // the way we display the voltage will be changed
  if (InVoltage < MinVoltage)  // if the voltage became below min voltage then indicator led will blink until input voltage becomes above min voltage
  {
    digitalWrite(Low_InVoltage_LED, HIGH);
    delay(500);
    digitalWrite(Low_InVoltage_LED, LOW);
    delay(500);
  } else if (InVoltage >= MinVoltage) {
    digitalWrite(Low_InVoltage_LED, LOW);
  }
  delay(250);
}
// char getData() {
//   if (Serial.available() > 0) {
    // char receivedChar = Serial.read();
//     newData = true;
//     return receivedChar;
//   }
// }

// void getData() {
//     static byte ndx = 0;
//     char endMarker = '\n';
//     char rc;
    
//     while (Serial.available() > 0 && newData == false) {
//         rc = Serial.read();

//         if (rc != endMarker) {
//             key[ndx] = rc;
//             ndx++;
//             if (ndx >= numChars) {
//                 ndx = numChars - 1;
//             }
//         }
//         else {
//             key[ndx] = '\0'; // terminate the string
//             ndx = 0;
//             newData = true;
//         }
//     }
// }
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
  voltage = (Current * refVoltage) / Mp_resolution;
  consumed_current = ((voltage - NoLoadVolt) / Sensitivity);
  if (consumed_current > MaxCurrent)  // if consumed current is above max current then a blinking LED will indicate this
  {
    digitalWrite(High_Consumed_I_LED, HIGH);
    delay(500);
    digitalWrite(High_Consumed_I_LED, LOW);
    delay(500);
  } else if (consumed_current < MaxCurrent) {
    digitalWrite(High_Consumed_I_LED, LOW);
  }

  Serial.println(consumed_current);
  delay(250);
}