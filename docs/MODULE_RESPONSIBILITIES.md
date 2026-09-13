# Module Responsibilities and Dependency Boundaries

本文档定义 `cylinder-detection` 的模块职责和允许的依赖方向。它不是简单描述“文件里现在写了什么”，而是约束 **以后新增代码应该放在哪里、哪些跨层依赖不允许重新出现**。

## 1. 总体分层

```text
main.cpp
  |
  v
Configuration / Runtime Control Context
  |
  v
HTTP Control Plane
(Server -> http/request_params + detect_request_mapper)
  |
  v
Application Orchestration
(Scheduler)
  |-------------------------------|
  v                               v
Hardware Acquisition              Detection Workers
(CameraThread / Slide)            (DetectThread / SingleDetectThread)
  |                               |
  v                               v
QueueManager                 DetectionPipeline
                                  |
                                  v
                        preprocess / HALCON / Slice
                             / YOLO / SAM
                                  |
                                  v
QueueManager -> GroupManager -> InspectionRepository -> SQLite
```

核心原则：

1. **HTTP 层不实现图像算法。**
2. **Scheduler 不实现 SQL，也不实现检测算法。**
3. **Worker 只负责线程/队列，不复制检测流水线。**
4. **DetectionPipeline 不读取 RuntimeState，不认识 HTTP 和数据库。**
5. **硬件 transport 不决定 GOOD/NG。**
6. **Repository 不控制相机、线程或队列。**
7. **共享状态只用于控制面，不能重新演化成 Service Locator。**
8. **业务含义通过名称和类型表达，不依赖数组位置、magic number 或 buffer 大小猜测。**

---

## 2. 程序入口与运行时上下文

| 文件 | 单一职责 | 不应承担的职责 |
| --- | --- | --- |
| `main.cpp` | CLI、配置加载、数据库 bootstrap、创建 `run_id`、启动 Server | HTTP 路由、算法、硬件控制、SQL 表映射 |
| `Core/Config.hpp/.cpp` | 读取/校验进程配置、解析部署路径 | HTTP 请求参数、模型推理、业务持久化 |
| `Core/runtime_state.hpp` | 控制面的进程级上下文：Config、当前任务快照、run_id | Worker 输出路径解析、算法执行 |
| `Core/detect_params.hpp` | 一次检测任务的纯参数 DTO | libevent、WinSock、SQLite、相机 SDK |
| `Core/detection_context.hpp` | 一次 process run 的输出根目录和 run_id | HTTP、数据库、线程生命周期 |

### RuntimeState 使用规则

`RuntimeState` 目前用于 Server 与 Scheduler 的控制面交接。`DetectThread`、`SingleDetectThread` 和 `DetectionPipeline` **不得再直接读取 `g_runtime_state`**。

Worker 所需的输出路径通过 `DetectionContext` 显式注入。新增算法模块也应遵循同样原则。

---

## 3. HTTP 控制面

### `Core/Server.hpp/.cpp`

Server 负责：

- libevent bind / route registration；
- API callback；
- response 输出；
- 调用 Scheduler、SingleDetectThread、CameraThread 的应用接口；
- 服务退出时的资源回收。

Server 不负责：

- JSON/query 类型转换细节；
- HTTP 字段到 `DetectParams` 的逐字段映射；
- 模型路径规则；
- 检测算法；
- SQL。

### `Core/http/request_params.hpp/.cpp`

职责：

```text
HTTP request
   |
   +-- JSON body
   +-- query string
   |
   v
统一 typed read-only parameter view
```

这里允许依赖 libevent / jsoncpp，但不允许依赖 Scheduler、Camera、TensorRT 或 SQLite。

### `Core/http/detect_request_mapper.hpp/.cpp`

职责：

```text
RequestParams + Config
        |
        v
DetectParams
        |
        +-- model path resolution
        +-- request validation
```

HTTP 字段兼容名、阈值范围、ROI/QW 参数合法性等应集中在 mapper，而不是重新散回 Server。

---

## 4. 多面检测应用层

| 文件 | 单一职责 |
| --- | --- |
| `Core/Scheduler.hpp/.cpp` | 多面任务生命周期、启动/停止顺序、Worker readiness gate |
| `Core/queue_manager.hpp/.cpp` | 原始帧/检测结果的并发传递与阻塞唤醒 |
| `Core/group_manager.hpp/.cpp` | 四面结果聚合、timeout、完成事件 |
| `Core/inspection_repository.hpp/.cpp` | 完成组到 SQLite 业务表的事务映射 |
| `Core/image_types.hpp` | 跨模块图像和结果 DTO |

### Scheduler 当前正确启动顺序

```text
QueueManager start
        |
        v
DetectThread x N start
        |
        v
wait all READY / detect startup failure
        |
        v
GroupManager + Repository callback
        |
        v
CameraThread / Slide hardware start
```

原因：模型/CUDA 初始化失败时，不应先移动滑台或产生无法处理的图像。

### Scheduler 可以做

- 构造 CameraThread、DetectThread、GroupManager；
- 显式构造 `DetectionContext`；
- 等待 Worker READY；
- 管理 stop/join/cleanup 顺序；
- 将 group-complete event 连接到 Repository。

### Scheduler 不可以做

- 拼 SQL；
- 实现 crop/HALCON/YOLO/SAM；
- 解析 HTTP；
- 操作 TensorRT buffer；
- 编码串口 transport。

---

## 5. 硬件层

| 文件 | 单一职责 |
| --- | --- |
| `Core/camera_thread.hpp/.cpp` | 相机 SDK 生命周期、单拍/四面采集、当前滑台业务时序 |
| `Core/slide_serial.hpp/.cpp` | 串口 handle、reader thread、按行收发 transport |

`SlideSerialClient` 不解释 GOOD/NG，也不管理 GroupManager。

当前 LOAD / GOTO_CAM / CAM_NEXT / UNLOAD 状态机仍在 CameraThread。如果协议继续增长，应演进为：

```text
CameraThread -> SlideController -> SlideSerialClient
               protocol           transport
```

不要继续把复杂协议状态堆到串口类或检测算法中。

---

## 6. Worker 与 DetectionPipeline

### `Core/detect_thread.hpp/.cpp`

多面 Worker 现在只负责：

- 一个 thread；
- startup READY/FAILED 信号；
- 从 Raw Queue 取 `ImageFrame`；
- 调 `DetectionPipeline::processFrame()`；
- 向 Result Queue 发布 `SingleImageResult`。

它不再直接持有 HALCON、SliceDetector、SAM，也不拼输出路径。

### `Core/single_detect_thread.hpp/.cpp`

只负责：

- 单图任务队列；
- submit / wait；
- stop / join；
- 调 `DetectionPipeline::configure()` 与 `processFile()`。

它不再维护第二套 preprocess/HALCON/YOLO/SAM 实现。

### `Core/detection_pipeline.hpp/.cpp`

这是当前唯一的检测主链实现：

```text
source image
    |
    v
GPU preprocess
    |
    v
HALCON center extraction
    |
    v
GPU slicing
    |
    v
TensorRT YOLO
    |
    v
SAM segmentation / measurement
    |
    v
SingleImageResult
```

职责还包括：

- CUDA device 选择；
- detector/SAM model cache；
- SAM engine -> ONNX fallback；
- 像素到毫米换算；
- Detection DTO 填充；
- 算法失败转换为 `processing_ok=false`。

Pipeline 不应：

- 读 `g_runtime_state`；
- push/pop QueueManager；
- 返回 HTTP response；
- 写 SQLite；
- 启停相机。

---

## 7. 预处理层

### `Core/detect/preprocess_options.hpp`

集中表达 ROI、QW mask、stripe removal、denoise 和保存选项。

新增调用点应使用：

```cpp
PreprocessOptions options;
preprocessImage(input, options, stream);
```

而不是继续扩展 20+ 个位置参数。

### `crop_image.h/.cpp`

`preprocessImage()` 是首选 API。

历史 `cropImage(...)` 长参数函数目前仅作为兼容 wrapper，内部转换为 `PreprocessOptions` 后调用统一实现。后续所有旧调用迁移完成后再删除 wrapper。

---

## 8. 算法层 `Core/detect/`

| 文件 | 职责 |
| --- | --- |
| `preprocess_options.hpp` | 预处理参数模型 |
| `crop_image.*` | GPU 预处理编排 |
| `StripeRemoval.*` | 条纹方向分析、频域抑制、可选去噪 |
| `HalconProcessor.*` | HALCON 候选中心提取和异步 overlay 保存 |
| `Slice.*` | 根据中心点生成 slice |
| `trtyolo_slice.*` | slice batch 推理、坐标还原、过滤、NMS |
| `sam.*` | YOLO box -> mask、过滤、几何度量、可视化 |
| `speedSam.*` | SAM encoder / decoder 业务编排 |
| `engineTRT.*` | TensorRT build/load/context/buffer/stream/tensor mapping |
| `config.h` | 历史 SAM 模型固定维度 |
| `logging.h` | TensorRT ILogger 实现 |
| `macros.h` | DLL/TRT 兼容宏 |
| `cuda_utils.h` | 历史 CUDA 宏，逐步淘汰 |

### 算法层硬规则

- 不 include `Server.hpp`；
- 不访问 RuntimeState；
- 不访问 Repository/SQLite；
- 不控制 CameraThread；
- CUDA/TensorRT 错误必须显式传播或转换，不依赖 `assert` 作为生产错误处理；
- 参数优先使用结构体而不是继续增加 positional arguments。

---

## 9. TensorRT 边界

`EngineTRT` 当前负责：

- ONNX build 或 engine deserialize；
- Runtime / Engine / ExecutionContext 生命周期；
- host/device buffer；
- CUDA copy stream；
- configured tensor name -> engine tensor index 映射。

### 规则

业务代码传入的 tensor name 是契约：

```text
SAM encoder output: image_embeddings
SAM decoder input : image_embeddings, point_coords, point_labels,
                    mask_input, has_mask_input
SAM decoder output: iou_predictions, low_res_masks
```

不得重新通过：

- TensorRT 枚举顺序；
- “第几个 tensor”；
- output buffer 大小；

去猜业务含义。

当前仍保留 `executeV2` binding-array 路径以避免静态重构改变部署行为。若未来统一 TensorRT 版本，可以单独验证 name/address based enqueue API。

---

## 10. 数据库层

| 文件 | 单一职责 |
| --- | --- |
| `Core/sqlite_helper.hpp/.cpp` | SQLite connection / prepared statement 的 RAII 与低层机制 |
| `Core/db_utils.hpp/.cpp` | schema bootstrap 和 run 注册 |
| `Core/inspection_repository.hpp/.cpp` | inspection domain persistence |

明确分层：

```text
SQLiteHelper         = database mechanism
DB Utils             = schema/bootstrap
InspectionRepository = domain persistence
```

新增业务 SQL 进入 Repository，不进入 Scheduler、Server 或 Worker。

下一次 schema 变化前应补正式 migration/version 机制，而不是继续只靠 `CREATE TABLE IF NOT EXISTS`。

---

## 11. 工具和工程文件

| 文件 | 职责 / 约束 |
| --- | --- |
| `Core/Utils/Log.hpp` | 项目统一轻量日志 |
| `Core/Utils/Common.hpp` | 无状态通用函数；禁止业务逻辑堆积 |
| `config.json` | 可移植运行时配置示例 |
| `mt.sln` | Visual Studio solution |
| `mt.vcxproj` | source + dependency properties |
| `mt.vcxproj.filters` | Visual Studio 逻辑目录 |
| `pytesttool/api_common.py` | API 测试共享 HTTP/config helper |
| `pytesttool/api_*.py` | 手工/冒烟 API 客户端 |

工程文件中新增模块必须同步登记，避免“Git 仓库里存在、Visual Studio 工程不编译”的隐性错误。

---

## 12. 允许的依赖方向

推荐：

```text
Server -> http/*
Server -> Scheduler / SingleDetectThread / CameraThread
Scheduler -> DetectThread / GroupManager / CameraThread / Repository
DetectThread -> DetectionPipeline
SingleDetectThread -> DetectionPipeline
DetectionPipeline -> detect/*
GroupManager -> Repository (through completion callback wiring)
Repository -> SQLiteHelper
CameraThread -> SlideSerialClient
```

避免：

```text
Algorithm -> Server
Algorithm -> RuntimeState
DetectionPipeline -> QueueManager
DetectionPipeline -> SQLite
SQLiteHelper -> Scheduler
Repository -> CameraThread
SlideSerialClient -> GroupManager
Config -> HTTP request
Worker -> global output/run state
```

出现新的跨层需求时，优先增加显式接口、context 或 callback，而不是直接 include 更高层文件或新增全局变量。
