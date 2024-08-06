// 定义模拟输入引脚
const int analogPin = A1;

// 定义电压和距离的对应关系
const float voltageMin = 0.0;
const float voltageMax = 5.0;
const float distanceMin = 20.0;
const float distanceMax = 45.0;

void setup() {
  // 初始化串口通信
  // Serial.begin(921600);  // 使用更高的波特率，以提高串口通信速度
  Serial.begin(57600);
}

void loop() {
  // 读取模拟电压值（范围为0到1023）
  int analogValue = analogRead(analogPin);
  
  // 将模拟值转换为电压值（范围为0.0到5.0）
  float voltage = analogValue * (voltageMax / 1023.0);
  
  // 将电压值转换为距离值（范围为20.0到45.0, 电压越大距离越小）
  float distance = map(voltage, voltageMin, voltageMax, distanceMax, distanceMin);
  
  // 输出距离值到串口监视器
  // Serial.print("Voltage: ");
  // Serial.print(voltage, 5);
  // Serial.print(" V, Distance: ");
  // Serial.print(distance, 5);
  // Serial.println(" mm");
  Serial.println(distance, 5);
  
  // 延迟最小时间以获得最高采样频率
  delay(10);  // 可选：使用最小的延迟时间
}

// 自定义map函数，将输入值从一个范围映射到另一个范围
float map(float x, float in_min, float in_max, float out_min, float out_max) {
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}




