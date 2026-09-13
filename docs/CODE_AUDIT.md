# Code Audit & Refactor Roadmap

本文件记录对 `cylinder-detection` 当前代码的第一轮工程审计。目标不是一次性重写全部代码，而是把问题按风险拆分，优先修复会导致 **换机器失效、任务状态错误、线程卡死、数据错配、部署困难** 的问题。

风险等级：

- **P0**：可能导致错误结果、死锁/永久等待、数据损坏或服务错误反馈；
- **P1**：强耦合、严重硬编码、资源生命周期与可维护性问题；
- **P2**：重复代码、命名、目录、测试与工程质量问题；
- **P3**：风格与长期优化。

---

## 1. 运行时配置与硬编码

### P0/P1：开发机绝对路径

位置：

- `config.json`
- `mt.vcxproj`

问题：历史配置直接写入 `E:\ydk\...`、`D:\Software\...`。这会使仓库在另一台机器上无法直接运行或编译。

当前处理：

- `config.json` 已改成相对路径；
- `Config` 会以配置文件所在目录为基准解析 `modelDir` / `uploadDir` / `dbPath`；
- **`mt.vcxproj` 暂未自动改写**，因为 CUDA、TensorRT、HALCON、相机 SDK、ThirdParty 的真实安装方式必须结合目标开发机验证。

推荐后续：

1. 建立 `Directory.Build.props` 或 `mt.props.example`；
2. 用环境变量或 MSBuild property 传入 `CUDA_PATH`、`TENSORRT_ROOT`、`THIRDPARTY_ROOT`；
3. 不要在 `.vcxproj` 中保存个人机器路径；
4. Release/Debug、Win32/x64 依赖配置统一。

---

## 2. `run_id` 生命周期

### P0：入口和 Scheduler 重复生成 `run_id`

历史行为：

- `main.cpp` 生成并注册一次 `run_id`；
- `Scheduler::start()` 又生成一次并覆盖全局状态。

风险：

- 数据库 `inspection_runs` 与实际 group 写入使用不同批次；
- 输出目录和数据库记录难以对应；
- 单图接口与多图接口可能落到不同 run 语义。

当前处理：

- `run_id` 只在 `main.cpp` 初始化一次；
- Scheduler 只消费 `g_server_state.run_id`。

长期建议：把 `RuntimeContext { run_id, config, paths }` 作为显式依赖注入，不再读取全局变量。

---

## 3. HTTP Server 责任过重

### P1：`Core/Server.cpp` 同时承担太多职责

当前包含：

- HTTP 协议；
- GET/POST 参数解析；
- DetectParams 组装；
- 模型路径拼接；
- Scheduler 生命周期；
- SingleDetectThread 生命周期；
- CameraThread 生命周期；
- 文件保存；
- 全局对象；
- API 返回序列化。

推荐拆分：

```text
Core/http/
  HttpServer
  HttpResponse
  RequestParser
Core/service/
  DetectionService
  CameraService
Core/runtime/
  RuntimeContext
Core/config/
  RuntimeConfig
```

目标：Server 层只负责“请求 -> DTO -> Service -> Response”。

---

## 4. 参数解析不完整

### P0：`device_id` 定义但服务端未真正解析

`DetectParams::device_id` 默认 `CAM-001`，测试脚本会发送 `device_id`，但当前 GET/POST parser 没有赋值。

结果：

- 多台设备写数据库时仍可能全部记录为 `CAM-001`；
- 日志与数据追踪失真。

建议：GET/POST 都解析 `device_id`，并做长度/字符校验。

### P1：SAM 文件名写死在 `DetectParams`

目前：

```text
SAM\SAM_encoder.engine
SAM\SAM_mask_decoder.engine
SAM\SAM_encoder.onnx
SAM\SAM_mask_decoder.onnx
```

建议移到配置：

```json
"models": {
  "samEncoder": "SAM/SAM_encoder.engine",
  "samDecoder": "SAM/SAM_mask_decoder.engine"
}
```

### P1：允许类别 `{0, 2}` 写死

位置：

- `Core/detect_thread.cpp`
- `Core/single_detect_thread.cpp`

风险：模型类别发生变化时，代码静默过滤错误类别。

建议配置：

```json
"allowedClasses": [0, 2]
```

---

## 5. GPU 设备硬编码

### P1：`cudaSetDevice(0)` / `cv::cuda::setDevice(0)`

位置：

- `Core/detect_thread.cpp`
- `Core/single_detect_thread.cpp`

当前配置已经增加 `gpuDevice`，但算法线程尚未全部改成消费该配置。

后续应做到：

- 所有 CUDA / OpenCV CUDA / TensorRT 初始化使用同一个 device；
- 线程启动时打印 device id；
- 多 GPU 场景明确 worker -> GPU 映射；
- 禁止一部分模块固定 0、一部分模块跟环境变量走。

---

## 6. Scheduler 生命周期

### P0：Scheduler 启动失败时 HTTP 仍可能返回成功

`api_control_add()` 获取了 `started`，但历史代码没有根据它决定返回值。

建议：

- `start()` 失败必须返回明确错误；
- `has_task` 在失败后恢复 false；
- 失败原因区分：camera init / model init / queue / db / serial。

### P1：Worker 模型加载是异步的

Scheduler 创建 worker 后立即把系统视为“启动成功”，但检测线程随后才加载 TensorRT/SAM。

风险：API 已返回成功，但 worker 实际几秒后才发现模型失败。

建议设计 `Worker::startAndWaitReady(timeout)`：

```text
CREATED -> INITIALIZING -> READY / FAILED -> STOPPING -> STOPPED
```

Scheduler 只有全部 worker READY 后才返回成功。

---

## 7. CameraThread

### P0：采集失败可能无限重试

多面采集 loop 中，`Snap()` 失败后持续 `continue`，只有外部 stop 才退出。

如果相机掉线，当前 group 可能永久卡在某一面。

建议：

- `maxCaptureRetries`；
- 每次 retry 指数退避；
- 超过阈值后产生“缺失面结果”或 abort group；
- 上报 camera unhealthy。

### P0：单张采集 timeout 参数没有贯穿

`captureSingleImage(int timeout_ms)` 接受 timeout，但工作线程内部固定调用 `captureSingleImageInternal(10000)`。

建议把 timeout 保存到 request state，并由 worker 使用实际传入值。

### P1：单图 request 使用 promise + bool 状态

当前同一时间只允许一个 single capture。设计本身可以接受，但应显式建模为一个 request object，避免未来扩展时 promise 被覆盖。

### P1：相机固定打开设备列表第 0 个设备

`initDevice()` 使用：

```text
GetDeviceInfo(0)
```

`device_id` 目前只是业务标签，并没有用于选择真实硬件。

建议增加：

- camera serial / hardware id；
- config 中指定 camera selector；
- 多相机时按 serial 精确匹配。

---

## 8. QueueManager 与背压

### P1：队列没有容量上限

`moodycamel::ConcurrentQueue` 当前没有业务层最大深度控制。

风险：检测速度低于采集速度时，内存持续增长，尤其 `ImageFrame` 含两张完整 `cv::Mat`。

建议：

- `maxRawQueueDepth`；
- `maxResultQueueDepth`；
- 采集侧明确 block / drop oldest / reject 策略；
- health API 暴露 high-water mark。

### P1：全局 QueueManager

`g_queue_manager` 让 Camera / Detect / Group 都依赖单例，测试困难。

长期建议 Scheduler 持有 QueueManager，并以引用传给组件。

---

## 9. GroupManager

### P1：完成判定重复扫描

`handleResult()` 内扫描一次 `all_received`，但不 finalize；`run()` 随后再次加锁扫描。

建议 `handleResult()` 返回 enum：

```text
Invalid / Duplicate / Accepted / Complete
```

然后在锁外直接 finalize。

### P1：timeout 缺失面通过伪 Detection(label=-1) 表示

这是业务状态和算法检测结果混在一起。

建议增加：

```cpp
enum class FaceStatus { Ok, Ng, Missing, Failed };
```

不要用“假缺陷框”表示系统错误。

---

## 10. DetectThread

### P0：HALCON 失败或 centers 为空时直接 `continue`

当前 worker 不向 result queue 推结果。

结果：GroupManager 只能等 timeout 才知道该面没回来。

建议：任何输入 frame 都应该得到一个结果事件：

```text
Success
NoFeature
InferenceFailed
PreprocessFailed
Cancelled
```

### P1：固定 sleep 作为初始化节奏

存在多处：

```text
sleep_for(2s)
sleep_for(1s)
sleep_for(1s)
```

这种方式不能保证资源真正 ready，只会增加启动时间。

建议删除，用真实初始化返回值和状态同步。

### P1：多 worker 各自加载完整模型

默认 4 个 DetectThread，每个持有独立 YOLO/SAM 模型。

风险：

- GPU 显存按 worker 倍增；
- 启动时间增加；
- TensorRT execution context / engine ownership没有明确区分。

需要根据 TensorRT 封装确认：

- engine 是否可共享；
- execution context 是否必须 per-thread；
- SAM encoder/decoder 是否可共享权重。

推荐架构：共享 immutable engine + per-worker execution context / stream。

### P1：CUDA stream 使用不一致

部分函数显式传 stream，部分调用没有传；单图路径中还存在 `cudaDeviceSynchronize()`。

风险：破坏异步 pipeline，造成全 device 同步。

建议统一 stream ownership，并减少 device-wide sync。

---

## 11. 单图与多图 Pipeline 重复

### P1：同一算法有两套实现

- `DetectThread::run()`
- `SingleDetectThread::worker()`

它们的 ROI、QW、stream、save、同步策略并不完全一致。

建议抽出：

```cpp
class DetectionPipeline {
public:
    PipelineResult process(const cv::Mat&, const DetectParams&, ProcessingContext&);
};
```

多图 worker 与单图 worker 只负责输入输出，不再复制算法过程。

---

## 12. SQLite

### P0：历史 UPSERT 先字符串拼 SQL 再 update/insert

第一轮已把 group / face 写入逐步改成 `ON CONFLICT`。

剩余建议：

- SQLiteHelper 增加 `queryPrepared()`；
- 所有动态参数都走 bind；
- transaction 引入 RAII，异常自动 rollback；
- 打开 `foreign_keys=ON` 应在每个连接设置，而不是只初始化连接；
- 根据并发需求设置 `busy_timeout` / WAL。

### P1：`PRAGMA foreign_keys = OFF` 创建 schema 后再 ON

SQLite 的 PRAGMA 是 connection-local。后续 `save_group_db()` 新开连接时未显式 ON，外键约束可能并未开启。

建议 SQLiteHelper 构造时执行：

```sql
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;
```

可评估：

```sql
PRAGMA journal_mode = WAL;
```

### P2：文件名拼写

`Core/sqlite_herpler.cpp` 应重命名为 `sqlite_helper.cpp`。

---

## 13. Build System

### P1：`.vcxproj` 同时 import CUDA 12.6 和 CUDA 11.8 props

工程中：

- `CUDA 12.6.props`
- `CUDA 11.8.props`
- target 又使用 CUDA 11.8

同时 TensorRT 路径写死在 `D:\Software\cu118\TensorRT`。

这是高风险构建配置。

建议明确唯一 CUDA baseline，例如：

```text
CUDA 11.8 + 对应 TensorRT + 对应 OpenCV CUDA
```

或升级整套依赖，不要混用 props。

### P1：ThirdParty 用 `..\ThirdParty`

仓库本身没有这些依赖，README 必须明确外部目录布局，长期最好：

- vcpkg / Conan 管理开源依赖；
- 商业 SDK 通过环境变量和 props 指定；
- CI 至少能做“头文件/纯逻辑”构建或静态检查。

---

## 14. 大小写与目录

### P2：`Detect/` vs `detect/`

代码 include 使用 `Detect/...`，真实目录是 `Core/detect/`。

Windows 默认不敏感，所以暂时可编译；迁移到 Linux/clang tooling 时会失败。

建议统一全仓库为：

```text
Core/detect/
#include "detect/..."
```

---

## 15. 测试脚本

### P1：`pytesttool/api_add.py` 与服务端参数历史不一致

脚本发送过：

```text
trt_engine_file
yolo_model_name
line_x1
line_y1
stripe_radius
labels
save_slices
```

而当前服务端主要读取：

```text
model_name
circle_x1
circle_y1
radius
...
```

这会让测试看起来“请求成功”，但大量参数被服务端静默忽略。

建议：

- 只保留服务端真实支持字段；
- 服务端对未知字段返回 warning/error；
- 把 `pytesttool` 改名为 `tools/api`，因为它并不是 pytest test suite。

---

## 16. README / 文档

第一轮已新增根 README，并用 Mermaid 表达：

- 系统组件关系；
- 四面采集/检测/入库 sequence；
- 配置与目录；
- 已知技术债。

建议后续把 `docs/architecture.md` 作为详细设计文档，README 只保留面向使用者的信息。

---

# 推荐重构顺序

## Phase 1：先保证“不会错”

1. HTTP 解析 `device_id`；
2. Scheduler 启动失败正确返回 API；
3. Camera capture retry 上限；
4. DetectThread 任何失败都生成 result event；
5. SQLite 每连接启用 foreign keys / busy timeout；
6. 单张采集 timeout 真正生效；
7. worker 初始化增加 READY/FAILED 状态。

## Phase 2：消灭硬编码

1. GPU device；
2. allowed classes；
3. SAM model paths；
4. camera hardware selector；
5. capture timeout / retry；
6. queue size；
7. HTTP body limit；
8. build dependency roots。

## Phase 3：去全局与去重复

1. `RuntimeContext`；
2. QueueManager 由 Scheduler 持有；
3. Camera/Slide/Detection interface；
4. 合并单图、多图 `DetectionPipeline`；
5. Server -> Service 分层。

## Phase 4：可测试与 CI

1. FakeCamera；
2. FakeSlide；
3. FakeDetector；
4. GroupManager 单测；
5. Config 单测；
6. SQLite integration test；
7. API parser test；
8. GitHub Actions 静态检查。

---

# 建议目标结构

```text
src/
├── app/
│   └── main.cpp
├── config/
│   ├── RuntimeConfig.hpp
│   └── RuntimeConfig.cpp
├── http/
│   ├── HttpServer.hpp
│   └── HttpServer.cpp
├── service/
│   ├── DetectionService.hpp
│   └── CameraService.hpp
├── pipeline/
│   ├── DetectionPipeline.hpp
│   ├── Preprocessor.hpp
│   ├── CenterExtractor.hpp
│   ├── ObjectDetector.hpp
│   └── Segmenter.hpp
├── hardware/
│   ├── ICamera.hpp
│   ├── MphCamera.hpp
│   ├── ISlide.hpp
│   └── SerialSlide.hpp
├── runtime/
│   ├── Scheduler.hpp
│   ├── GroupManager.hpp
│   └── QueueManager.hpp
├── storage/
│   ├── InspectionRepository.hpp
│   └── SqliteInspectionRepository.hpp
└── model/
    └── Types.hpp
```

该结构不建议一次性“大搬家”。应在测试与接口边界先建立后逐模块迁移。
