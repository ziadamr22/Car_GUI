
#define RED_PIN       6    //PWM pin   used common cathod RGB if u used common anode then change TurnON_RGB to 0 and TurnON_RGB to 255
#define GREEN_PIN     5    //PWM pin
#define BLUE_PIN      3    //PWM pin
#define TurnON_RGB    255
#define TurnOFF_RGB   0
 

void setup() {
  Serial.begin(9600);
  pinMode(RED_PIN,OUTPUT);
  pinMode(GREEN_PIN,OUTPUT);
  pinMode(BLUE_PIN,OUTPUT);
  
}

void loop() {
  // put your main code here, to run repeatedly:

}







void RGB_SpeedIndicator(int MotorSpeed)
{ 
  MotorSpeed = analogRead(pot);
  MotorSpeed = map(MotorSpeed,0,1023,0,2);
  if(MotorSpeed < 1)
  {
    analogWrite(GREEN_PIN,TurnON_RGB); 
    analogWrite(RED_PIN,TurnOFF_RGB); 
    analogWrite(BLUE_PIN,TurnOFF_RGB); 
  }
  else if(MotorSpeed < 2)
  {
    analogWrite(RED_PIN,TurnON_RGB); 
    analogWrite(GREEN_PIN,TurnON_RGB);
    analogWrite(BLUE_PIN,TurnOFF_RGB); 
   }
  
  else{
    analogWrite(RED_PIN,TurnON_RGB); 
    analogWrite(GREEN_PIN,TurnOFF_RGB);
    analogWrite(BLUE_PIN,TurnOFF_RGB); 
  }
  
}
