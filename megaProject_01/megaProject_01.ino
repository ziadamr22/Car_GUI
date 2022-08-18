#include <Keypad.h>
#define RED_PIN              11    //PWM pin   used common cathod RGB if u used common anode then change TurnON_RGB to 0 and TurnON_RGB to 255
#define GREEN_PIN            10    //PWM pin
#define BLUE_PIN             9     //PWM pin
#define Enable_Pin           3     //PWM pin
#define C_Motor1_dir_A1      2
#define C_Motor1_dir_A2      4
#define C_Motor2_dir_A1      7
#define C_Motor2_dir_A2      8 
#define C_Motor1_Speed_A3    3     //PWM pin
#define C_Motor2_Speed_A3    6      //PWM pin   
#define pot                  A0
//#define C_Motor_pot          A1 
#define joyX                 A2
#define joyY                 A3
#define TurnON_RGB           255
#define TurnOFF_RGB          0
#define Low_Speed            50 
#define Med_Speed            150
#define High_Speed           255
int MotorSpeed;
int X_value,Y_value;
int pwmOutput;
char key;
const byte ROWS = 4; //four rows
const byte COLS = 4; //four columns
char keys[ROWS][COLS] = {
  {'1','2','3','A'},
  {'4','5','6','B'},
  {'7','8','9','C'},
  {'*','0','#','D'}
};                        
byte rowPins[ROWS] = {0,1,5,6}; //connect to the row pinouts of the keypad
byte colPins[COLS] = {7,8,12,13}; //connect to the column pinouts of the keypad

Keypad keypad = Keypad( makeKeymap(keys), rowPins, colPins, ROWS, COLS );  // struct to pass (keys , rowpins,rows and cols) values to it 


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
  

  
}

void loop() {
  //X_value = analogRead(joyX);
  //Y_value = analogRead(joyY);
   //pwmOutput = analogRead(pot); 
   //pwmOutput = map(pwmOutput, 0, 1023, 0 , 255); 
   //analogWrite(C_Motor1_Speed_A3, pwmOutput); // Send PWM signal to L298N Enable pin
   
   key = keypad.getKey();
   if (key) 
   {  
      if(key == '2')
      {
        Forward();
      }
      else if(key == '8')
      {
        Backward(); 
      }
      else if(key == '6')
      {
        Right();
      }
      else if(key == '4')
      {
        Left(); 
      }
      else if(key == '5')
      {
        Stop();
      }
    
   }
   
  MotorSpeed = analogRead(pot);
  int Speed = SetMotor_Speed(MotorSpeed);
  RGB_SpeedIndicator(Speed);
 
  

}







void RGB_SpeedIndicator(int MotorSpeed)
{ 
  if(MotorSpeed == 0)
  {
    analogWrite(GREEN_PIN,TurnON_RGB); 
    analogWrite(RED_PIN,TurnOFF_RGB); 
    analogWrite(BLUE_PIN,TurnOFF_RGB); 
  }
  else if(MotorSpeed == 1)
  {
    analogWrite(RED_PIN,TurnON_RGB); 
    analogWrite(GREEN_PIN,TurnON_RGB);
    analogWrite(BLUE_PIN,TurnOFF_RGB); 
   }
  
  else if (MotorSpeed == 2){
    analogWrite(RED_PIN,TurnON_RGB); 
    analogWrite(GREEN_PIN,TurnOFF_RGB);
    analogWrite(BLUE_PIN,TurnOFF_RGB); 
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
