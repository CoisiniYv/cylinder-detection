# Module Responsibilities and Dependency Boundaries

本文档定义 `cylinder-detection` 的模块职责与依赖边界。目标不是描述“某个文件现在恰好写了什么”，而是约束 **以后什么代码应该放在哪里、什么代码不应该放在哪里**。

## 1. 总体分层

```text
main.cpp
  |
  v
Configuration / Runtime Context
  |
  v
HTTP Control Plane (Server)
  |
  v
Application Orchestration (Scheduler)
  |---------------------------|
  v                           v
Hardware Acquisition          Detection Workers
(Camera / Slide)              (DetectThread / SingleDetectThread)
  |                           |
  v                           v
QueueManager              Algorithm Components
  |                       (crop/HALCON/slice/YOLO/SAM)
  |                           |
  +-------------> GroupManager
                    |
                    v
             InspectionRepository
                    |
                    v
                  SQLite
```

核心原则：

1. **HTTP 层不实现算法。**
2. **Scheduler 不实现数据库表映射，也不实现图像算法。**
3. **算法模块不读取 HTTP 请求，不管理服务生命周期。**
4. **硬件传输层不决定业务 GOOD/NG。**
5. **数据库层不控制相机和线程。**
6. **共享状态只保存进程级上下文，不作为随手可读写的全局参数仓库。**

---

## 2. 程序入口与运行时上下文

| 文件 | 单一职责 | 不应承担的职责 |
| --- | --- | --- |
| `main.cpp` | CLI、加载配置、初始化数据库、创建 `run_id`、启动 Server | HTTP 路由、算法、相机控制、SQL 表映射 |
| `Core/Config.hpp/.cpp` | 读取/校验静态进程配置、解析相对路径 | 请求级检测参数、模型推理、数据库业务操作 |
| `Core/runtime_state.hpp` | 保存进程级运行上下文：Config、当前任务快照、run_id | 算法参数计算、线程实现、持久化逻辑 |
| `Core/detect_params.hpp` | 定义一次检测任务的纯数据参数 | libevent、WinSock、SQLite、硬件 SDK |

### RuntimeState 使用规则

`RuntimeState` 是控制面的共享上下文，不应该成为任意模块的 Service Locator。新增代码优先通过构造函数或函数参数显式传递依赖。

当前 `DetectThread` / `SingleDetectThread` 仍会读取 `g_runtime_state` 获取输出目录和 `run_id`，这是后续应继续移除的耦合点。

---

## 3. HTTP 控制面

| 文件 | 单一职责 |
| --- | --- |
| `Core/Server.hpp` | 对外暴露最小化 Server 生命周期接口 |
| `Core/Server.cpp` | libevent HTTP 绑定、请求解析、响应序列化、API 路由、应用服务触发 |

`Server.cpp` 可以知道 `Scheduler`、`SingleDetectThread`、`CameraThread` 的应用接口，但不应该直接实现检测流程本身。

后续推荐继续拆分：

```text
Server.cpp
├── HTTP transport / routing
├── RequestParams parser        -> http_request_params.*
├── DetectParams mapping        -> detect_request_mapper.*
└── response serialization      -> api_serialization.*
```

当前不强制一次拆完，避免在没有集成测试时造成大面积行为变化。

---

## 4. 多面检测应用层

| 文件 | 单一职责 |
| --- | --- |
| `Core/Scheduler.hpp/.cpp` | 多面检测 pipeline 的生命周期编排和停止顺序 |
| `Core/queue_manager.hpp/.cpp` | 原始帧和检测结果的线程安全传递、阻塞/唤醒语义 |
| `Core/group_manager.hpp/.cpp` | 四面结果聚合、超时策略、组完成事件 |
| `Core/inspection_repository.hpp/.cpp` | 将完成组映射到 SQLite 业务表并事务写入 |
| `Core/image_types.hpp` | 跨模块传递的图像、元信息和检测结果 DTO |

### Scheduler 允许做什么

- 创建/持有 CameraThread、DetectThread、GroupManager；
- 控制启动、停止、join 顺序；
- 将 GroupManager 完成事件连接到 Repository。

### Scheduler 不允许做什么

- 拼 SQL；
- 创建业务表；
- 实现 crop / HALCON / YOLO / SAM；
- 解析 HTTP 参数；
- 编码滑台协议指令。

---

## 5. 硬件层

| 文件 | 单一职责 |
| --- | --- |
| `Core/camera_thread.hpp/.cpp` | 相机 SDK 生命周期、采集循环、一次/四面采集时序，并协调滑台动作 |
| `Core/slide_serial.hpp/.cpp` | 串口句柄、reader thread、按行收发；只负责 transport |

### 边界说明

`SlideSerialClient` 不解释业务流程；LOAD / GOTO_CAM / CAM_NEXT / UNLOAD 的流程编排目前属于 `CameraThread`。

如果滑台协议继续扩大，应新增 `SlideController`：

```text
CameraThread -> SlideController -> SlideSerialClient
               protocol           transport
```

而不是继续把更多协议常量和状态机塞入 `CameraThread`。

---

## 6. 检测 Worker 层

| 文件 | 单一职责 |
| --- | --- |
| `Core/detect_thread.hpp/.cpp` | 多面 pipeline 中一个 GPU worker 的线程与模型实例生命周期 |
| `Core/single_detect_thread.hpp/.cpp` | 单图 API 的串行任务队列、模型缓存与任务完成同步 |

两个 Worker 当前仍重复实现大量：

```text
preprocess -> HALCON -> slice -> YOLO -> SAM -> metrics/result
```

长期应抽取无线程语义的 `DetectionPipeline`：

```text
DetectThread ------>
                    DetectionPipeline::process(input, params, context)
SingleDetectThread ->
```

Worker 只负责线程、排队、取消和错误转换；Pipeline 只负责一次图像处理。

在真实数据验证前不建议直接大规模合并，因为当前单图/多图在保存命名和 QW 使用方式上仍有差异。

---

## 7. 算法层 `Core/detect/`

| 文件 | 职责 |
| --- | --- |
| `crop_image.h/.cpp` | GPU 图像预处理入口：ROI、条纹处理、QW mask、可选保存 |
| `StripeRemoval.h/.cpp` | 频域条纹方向检测、频域抑制和可选 CUDA 去噪 |
| `HalconProcessor.h/.cpp` | HALCON 中心区域提取 + 异步骨架/overlay 保存 |
| `Slice.h/.cpp` | 基于中心点生成 CPU/GPU slice |
| `trtyolo_slice.hpp/.cpp` | slice batch 推理适配、坐标还原、过滤和 NMS |
| `sam.h/.cpp` | YOLO box -> SAM mask、几何度量、过滤与可视化 |
| `speedSam.h/.cpp` | SAM encoder/decoder 编排 |
| `engineTRT.h/.cpp` | TensorRT engine 构建/反序列化、buffer/context/stream 管理 |
| `config.h` | 历史 SAM 固定模型维度常量 |
| `logging.h` | TensorRT 示例风格 ILogger 实现 |
| `macros.h` | DLL/TRT 兼容宏 |
| `cuda_utils.h` | 历史 CUDA 检查宏；应逐步替换为异常/统一错误策略 |

### 算法层规则

- 不依赖 `Server.hpp`；
- 不访问 `g_runtime_state`；
- 不拼接业务数据库路径；
- 不控制 CameraThread；
- 参数尽量通过结构体显式传入，而不是继续增加 20+ 个位置参数。

---

## 8. 数据库层

| 文件 | 单一职责 |
| --- | --- |
| `Core/sqlite_helper.hpp/.cpp` | SQLite connection/prepared statement 的 RAII 与低层 API |
| `Core/db_utils.hpp/.cpp` | 数据库 schema 初始化和 run 注册 |
| `Core/inspection_repository.hpp/.cpp` | inspection 业务实体的读写映射 |

三层不能混淆：

```text
SQLiteHelper        = database mechanism
DB Utils            = schema/bootstrap
InspectionRepository= domain persistence
```

新增业务 SQL 应优先进入 Repository，而不是 Scheduler/Server。

---

## 9. 工具与工程文件

| 文件 | 职责 / 约束 |
| --- | --- |
| `Core/Utils/Log.hpp` | 项目统一轻量日志宏 |
| `Core/Utils/Common.hpp` | 无状态、跨模块通用的小工具；禁止堆业务逻辑 |
| `config.json` | 运行时部署配置示例 |
| `mt.sln` | Visual Studio solution |
| `mt.vcxproj` | MSBuild 源文件/第三方依赖配置 |
| `mt.vcxproj.filters` | Visual Studio 逻辑目录 |
| `.gitignore` | 构建产物、IDE 文件、输出数据排除 |
| `.vscode/settings.json` | 编辑器文件关联；不应存机器绝对路径 |
| `pytesttool/*.py` | HTTP API 手工/冒烟测试工具，不是服务业务实现 |

---

## 10. 建议依赖方向

允许：

```text
Server -> Scheduler -> Camera/Workers/GroupManager -> Repository
Worker -> detect/*
Repository -> SQLiteHelper
CameraThread -> SlideSerialClient
```

避免：

```text
Algorithm -> Server
Algorithm -> global RuntimeState
SQLiteHelper -> Scheduler
SlideSerialClient -> GroupManager
Repository -> CameraThread
Config -> HTTP request
```

如果未来新增模块，优先检查它是否遵循这个单向依赖方向。
