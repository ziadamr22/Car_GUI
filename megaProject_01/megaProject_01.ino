// #include <Keypad.h>
#define RED_PIN              11    //used common cathod RGB if u used common anode then change TurnON_RGB to 0 and TurnON_RGB to 255
#define GREEN_PIN            10    
#define BLUE_PIN             9     
#define C_Motor1_dir_A1      2
#define C_Motor1_dir_A2      4
#define C_Motor2_dir_A1      7
#define C_Motor2_dir_A2      8 
#define C_Motor1_Speed_A3    3     //PWM pin
#define C_Motor2_Speed_A3    6     //PWM pin   
#define Low_InVoltage_LED    5
//#define HIGH_InVoltage_LED   12
#define pot                  A0
//#define C_Motor_pot          A1 
#define joyX                 A2
#define joyY                 A3
#define VoltageSensor        A4
#define TurnON_RGB           HIGH
#define TurnOFF_RGB          LOW
#define Low_Speed            50 
#define Med_Speed            150
#define High_Speed           255
#define Mp_resolution        1023.0
#define factor               0.2           // this voltage sensor consists of a voltage divider circuit and its factor =R2/(R1+R2) = 7.5k/(30k+7.5k) = 0.2 where R2 is connected in parallel with Vout 
#define refVoltage           5.0           // arduino voltage
#define MinVoltage           15
//#define MaxVoltage           24
int MotorSpeed;
int X_value,Y_value;
int pwmOutput;
int Voltage;
int key;
boolean newData = false;
const byte ROWS = 4; //four rows
const byte COLS = 4; //four columns
// char keys[ROWS][COLS] = {
//   {'1','2','3','A'},
//   {'4','5','6','B'},
//   {'7','8','9','C'},
//   {'*','0','#','D'}
// };                        
byte rowPins[ROWS] = {0,1,5,6}; //connect to the row pinouts of the keypad
byte colPins[COLS] = {7,8,12,13}; //connect to the column pinouts of the keypad

// Keypad keypad = Keypad( makeKeymap(keys), rowPins, colPins, ROWS, COLS );  // struct to pass (keys , rowpins,rows and cols) values to it 

void setup() {
  Serial.begin(9600);
  // initializing indicator pins
  pinMode(RED_PIN,OUTPUT);
  pinMode(GREEN_PIN,OUTPUT);
  pinMode(BLUE_PIN,OUTPUT);
  // initializing Motor direction pins
  pinMode(C_Motor1_dir_A1,OUTPUT);
  pinMode(C_Motor1_dir_A2,OUTPUT);
  pinMode(C_Motor2_dir_A1,OUTPUT);
  pinMode(C_Motor2_dir_A2,OUTPUT);
  // initializing Motor speed pins
  pinMode(C_Motor1_Speed_A3,OUTPUT);
  pinMode(C_Motor2_Speed_A3,OUTPUT);
  // initializing input voltage indicator pins
  pinMode(Low_InVoltage_LED,OUTPUT);
}

void loop() {
  //X_value = analogRead(joyX);
  //Y_value = analogRead(joyY);
   //pwmOutput = analogRead(pot); 
   //pwmOutput = map(pwmOutput, 0, 1023, 0 , 255); 
   //analogWrite(C_Motor1_Speed_A3, pwmOutput); // Send PWM signal to L298N Enable pin
   
  //  key = keypad.getKey();
  key = String(getData()).toInt();
  // Voltage = analogRead(VoltageSensor);
  // Voltage_Sensor(Voltage);
  

  RGB_SpeedIndicator(key);
   if (key) 
   {  
      if(key == 3) //Signal sent from python to move forward = 3
      {
        Forward();
      }
      else if(key == 4) //Signal sent from python to move backward = 4
      {
        Backward(); 
      }
      else if(key == 5) //Signal sent from python to move right = 5
      {
        Right();
      }
      else if(key == 6) //Signal sent from python to move left = 6
      {
        Left(); 
      }
      else if(key == 7) //Signal sent from python to  stop = 7
      {
        Stop();
      }
    
   }
   //analogWrite(C_Motor_Speed_A3, pwmOutput); // Send PWM signal to L298N Enable pin
  //  key = keypad.getKey();
  //  if (key) 
  //  { 
  //   SetMotor_Dir(key);
  //  }   
  MotorSpeed = analogRead(pot);
  int Speed = SetMotor_Speed(MotorSpeed);  
}
void RGB_SpeedIndicator(int MotorSpeed)
{ 
  if(MotorSpeed == 0)
  {
    digitalWrite(GREEN_PIN,TurnON_RGB); 
    digitalWrite(RED_PIN,TurnOFF_RGB); 
    digitalWrite(BLUE_PIN,TurnOFF_RGB); 
  }
  else if(MotorSpeed == 1)
  {
    digitalWrite(RED_PIN,TurnOFF_RGB); 
    digitalWrite(GREEN_PIN,TurnOFF_RGB);
    digitalWrite(BLUE_PIN,TurnON_RGB); 
   }
  
  else if (MotorSpeed == 2){
    digitalWrite(RED_PIN,TurnON_RGB); 
    digitalWrite(GREEN_PIN,TurnOFF_RGB);
    digitalWrite(BLUE_PIN,TurnOFF_RGB); 
  }
  
}


int SetMotor_Speed(int Pot_Value)
{
  int Speed;
  Speed = map(Pot_Value,0,1023,0,2);
  if(Speed == 0)
  {
    analogWrite(C_Motor1_Speed_A3,Low_Speed);
    analogWrite(C_Motor2_Speed_A3,Low_Speed);
  }
  else if(Speed == 1)
  {
    analogWrite(C_Motor1_Speed_A3,Med_Speed);
    analogWrite(C_Motor2_Speed_A3,Med_Speed);
  }
  
  else if (Speed == 2)
  {
    analogWrite(C_Motor1_Speed_A3,High_Speed);
    analogWrite(C_Motor2_Speed_A3,High_Speed);
  }
  
  return Speed;
}

//void SetMotor_Dir(char key)
//{
    //X_value = map(X_value,0,1023,0,3);                  here,will replace joystick with keypad and below condition based on keypad    
    //Y_value = map(Y_value,0,1023,0,3);
//}

void Forward()
{                             
    digitalWrite(C_Motor1_dir_A1, HIGH);        //  Motor1 moves forward
    digitalWrite(C_Motor1_dir_A2, LOW);
    digitalWrite(C_Motor2_dir_A1, HIGH);        //  Motor2 moves forward
    digitalWrite(C_Motor2_dir_A2, LOW);
    delay(20);
}

void Backward()
{                                              
    digitalWrite(C_Motor1_dir_A1, LOW);          //  Motor1 moves backward
    digitalWrite(C_Motor1_dir_A2, HIGH);
    digitalWrite(C_Motor2_dir_A1, LOW);          //  Motor2 moves backward
    digitalWrite(C_Motor2_dir_A2, HIGH);
    delay(20);
}

void Right()
{                                              
    digitalWrite(C_Motor1_dir_A1, LOW);            //  Motor1 moves backward
    digitalWrite(C_Motor1_dir_A2, HIGH);
    digitalWrite(C_Motor2_dir_A1, HIGH);           //  Motor2 moves forward
    digitalWrite(C_Motor2_dir_A2, LOW);
    delay(20);
}

void Left()
{                                               
    digitalWrite(C_Motor1_dir_A1, HIGH);          //  Motor1 moves forward     
    digitalWrite(C_Motor1_dir_A2, LOW);          
    digitalWrite(C_Motor2_dir_A1, LOW);           //  Motor2 moves backward
    digitalWrite(C_Motor2_dir_A2, HIGH);
    delay(20);
}

void Stop()
{                             
    digitalWrite(C_Motor1_dir_A1, LOW);          // to stop motion
    digitalWrite(C_Motor1_dir_A2, LOW);
    digitalWrite(C_Motor2_dir_A1, LOW);
    digitalWrite(C_Motor2_dir_A2, LOW);
    delay(20);
}

void Voltage_Sensor(int Voltage)
{
    float mapped_voltage = 0.0;
    float InVoltage = 0.0;
    mapped_voltage = (Voltage * refVoltage)/Mp_resolution;
    InVoltage =  mapped_voltage / factor;
    if(InVoltage < MinVoltage )                       // if the voltage became below min voltage then indicator led will blink until input voltage becomes above min voltage 
    {
      digitalWrite(Low_InVoltage_LED,HIGH);
      delay(500); 
      digitalWrite(Low_InVoltage_LED,LOW);
      delay(500);
    }
    else if(InVoltage >= MinVoltage)
    {
      digitalWrite(Low_InVoltage_LED,LOW);
    }
    Serial.println(InVoltage);                      // the way we display the voltage will be changed
    delay(250);
}
char getData() {
    if (Serial.available() > 0) {
        char receivedChar = Serial.read();
        newData = true;
    return receivedChar;
    }
}

