# 串口通信手册（滑台控制工程）

本文档基于当前工程代码整理，作为后续联调与优化的通信基线。除明确说明外，接口格式不得擅自变更。

## 1. 总览

- 上位机控制口：UART1（ASCII 文本命令）
- 设备总线：UART2（内部设备通道，X_V2 二进制协议 + Modbus-RTU）

## 2. UART1 上位机控制口

### 2.1 串口参数

- 端口：USART1
- 波特率：115200
- 数据位：8
- 停止位：1
- 校验：None
- 流控：None

### 2.2 帧格式（ASCII）

```
<id>,<dir>,<steps>,<dly>\r\n
```

- 行结束：`\r` 或 `\n`（CR/LF 任一即可，CRLF 亦可）
- 行长度：最大 63 字符，超长会丢弃并重置接收
- 逗号分隔，数字之间不要出现空格

字段定义：

- `id`：轴 ID（推荐约定：0=X，1=Y，2=Z）
- `dir`：功能/方向码（见 2.3）
- `steps`：步数（对定位运动有效）
- `dly`：速度相关参数（单位 us，值越小速度越高）

### 2.3 命令定义（dir 取值）

| dir | 功能 | 参数说明 | 典型返回 |
|---|---|---|---|
| 0 | 定位运动（向 Min） | `steps` 步数，`dly` 速度(us) | `Start Ramp` / `NoMove` |
| 1 | 定位运动（向 Max） | `steps` 步数，`dly` 速度(us) | `Start Ramp` / `NoMove` |
| 8 | 回零（Home） | `dly` 速度(us)，`steps` 忽略 | `Start Home` |
| 9 | 对中（Align） | `dly` 速度(us)，`steps` 忽略 | `Start Align` |
| 20 | 流程：LOAD | 参数忽略 | `Start LOAD` / `Flow Busy` |
| 21 | 流程：GOTO_CAM | 参数忽略 | `Start GOTO_CAM` / `Flow Busy` |
| 22 | 流程：UNLOAD | 参数忽略 | `Start UNLOAD` / `Flow Busy` |
| 23 | 流程：CAM_NEXT | 参数忽略 | `Cam Next` |
| 其他 | 非法命令 | — | `Bad Cmd` |

### 2.4 返回信息（同步/异步）

启动/状态：

- `Calibrating...`
- `CAL <ok> X_R=<...> X_M=<...> Y_R=<...>`
- `Slide Ready`

错误/解析：

- `Parse Err`：命令解析失败
- `NoMove`：运动被限制为 0 步（安全/行程保护）
- `Flow Busy`：流程正在执行

运动与流程事件（异步上报）：

- 轴事件：`M<id> POS_DONE | HOME_DONE | ALIGN_DONE | STOP_MIN | STOP_MAX | STOP`
- 流程结束：`FLOW LOAD DONE` / `FLOW GOTO_CAM DONE` / `FLOW UNLOAD DONE`
- 开关触发：`SW X_L | X_M | X_R | Y_L | Y_R | Z_L | Z_R`

### 2.5 示例

单轴定位（X 轴向 Max 2000 步，速度 300us）：

```
0,1,2000,300
```

回零（Y 轴回零，速度 500us）：

```
1,8,0,500
```

启动流程（LOAD）：

```
0,20,0,0
```

### 2.6 约束与注意事项

- 固件会根据标定/限位自动裁剪步数，可能返回 `NoMove`。
- `dir=99` 的 E-STOP 在固件端未实现，现阶段会返回 `Bad Cmd`。
- UART1 接口格式为当前对外基线，后续优化应保持该格式不变。
- `id` 轴号建议遵循 0/1/2（X/Y/Z），固件未对 `id` 越界做统一错误返回。
- `dly` 为速度相关参数（us），实际速度曲线与轴实现有关，上位机需按“速度档位”调参。
- 协议无全局“忙/空闲”响应（除流程类 `Flow Busy`），上位机建议做发送节流或等待事件/状态回报后再发下一条。
- 解析为整数，暂不支持小数或十六进制输入；行长超限会丢弃。

### 2.7 C++ 调用方式（对应 aaaa.py 的调用逻辑）

本节描述 C++ 侧需要实现的调用方式与行为约定，等价于 `aaaa.py` 的功能（列出串口、连接、发送、轮询接收）。此处给出 **接口形态与行为约定**，不提供未验证的可编译示例代码。

#### 2.7.1 功能映射（aaaa.py -> C++）

- `/ports`：列出可用串口  
  - C++：`ListPorts()` 返回端口名列表（如 `COM7`、`COM11`）
- `/conn`：打开指定串口  
  - C++：`Open(port, 115200, timeout_ms)`  
  - 行为：打开前若已打开需先关闭
- `/send`：发送一行命令  
  - C++：`WriteLine("id,dir,steps,dly")`  
  - 行为：发送时自动追加 `\n`（或 `\r\n`）
- `/poll`：拉取接收缓存  
  - C++：后台线程读取串口行文本并写入队列  
  - 上层调用：`DrainLines()` 返回并清空队列

#### 2.7.2 推荐接口形态（C++ 头文件级描述）

```cpp
struct SerialConfig {
  std::string port;   // "COM11" / "/dev/ttyUSB0"
  int baud = 115200;
  int timeout_ms = 100;  // 读超时
};

class SlideSerialClient {
public:
  bool Open(const SerialConfig& cfg);
  void Close();
  bool IsOpen() const;

  // 发送一行命令，内部自动追加 '\n'
  bool WriteLine(const std::string& line);

  // 非阻塞拉取已接收的完整行（不含换行符）
  std::vector<std::string> DrainLines();

  // OS 相关：枚举可用串口
  static std::vector<std::string> ListPorts();
};
```

#### 2.7.3 行为约定

- 发送命令时必须保持 UART1 既定格式：`<id>,<dir>,<steps>,<dly>`  
- 写入串口时追加行结束（推荐 `\n`）  
- 接收端以 `\r` 或 `\n` 作为一行结束  
- 建议采用独立接收线程，将读取到的行放入线程安全队列，供上层轮询拉取  

#### 2.7.4 未提供可编译示例代码的说明

用户要求示例代码需经过单独测试后再提供。本文档仅给出 C++ 调用方式与行为约定，后续如需示例代码，需在目标系统与真实/仿真串口环境下验证后再输出。

## 3. UART2 设备总线（内部通道）

### 3.1 串口参数

- 端口：USART2
- 波特率：115200
- 数据位：8
- 停止位：1
- 校验：None
- 流控：None
- 备注：DMA + 中断，固件内部使用

### 3.2 协议 A：X_V2 电机驱动（二进制）

通用帧格式：

```
[addr][func][payload...][0x6B]
```

- 固定校验/终止字节：`0x6B`
- 多数指令只发送不接收
- 速度/位置类参数常按 10 倍放大后发送（0.1 单位）

多电机指令（MMCL）：

```
[addr][0xAA][len_hi][len_lo][MMCL_cmd...][0x6B]
```

系统参数读取返回示意：

```
[addr][0x37][sign][data_u32][0x6B]
```

完整命令列表与参数说明见：`Inc/X_V2.h`、`Src/X_V2.c`。

### 3.3 协议 B：Modbus-RTU（ERG32/Jodell）

- 功能码：0x03（读保持寄存器）、0x10（写多个保持寄存器）
- CRC16（Modbus）校验
- 高层封装接口：`Erg32_*`、`Jodell_*`

主要头文件：`Inc/erg_modbus_rtu.h`、`Inc/erg32.h`、`Inc/Jodell_Gripper.h`。

## 4. 版本与兼容性

- 本文档以当前固件实现为准。
- UART1 对外命令格式为稳定接口，后续优化需保持兼容。
- UART2 为内部设备通道，外部主机不建议直接接入。

