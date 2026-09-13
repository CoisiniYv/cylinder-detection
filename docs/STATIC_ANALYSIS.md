# Static Analysis Review

本文档记录对当前 `refactor/runtime-config-readme` 分支的静态代码审查。审查范围包含程序入口、Core 全部 C++ 源文件/头文件、`Core/detect` 算法层、Visual Studio 工程文件以及 `pytesttool` 脚本。

> 静态审查不能代替 Windows + 相机 + 滑台 + HALCON + CUDA/TensorRT 的真实编译与硬件回归。标记为“需实机验证”的项目没有仅凭代码猜测直接改变算法语义。

## 风险级别

- **P0**：可能卡死、丢任务、产生错误状态或数据损坏；
- **P1**：明显资源/并发/架构耦合风险；
- **P2**：可读性、重复代码、接口设计和构建维护问题；
- **P3**：命名、风格和长期整理。

---

## 本轮已修复

| 项目 | 风险 | 处理 |
| --- | --- | --- |
| QueueManager 条件变量存在 lost wake-up 窗口 | P0/P1 | 增加显式 item counter 和 predicate wait，停止时统一唤醒 |
| Scheduler 同时负责线程编排和 SQL/事务映射 | P1 | 新增 `InspectionRepository`，Scheduler 只连接生命周期与完成回调 |
| face_id 查询使用字符串拼 SQL | P1/P2 | 改为 prepared SELECT + bind + `stepInt64()` |
| SQLiteHelper 构造中 PRAGMA 抛异常可能泄漏连接 | P1 | 构造失败显式关闭 connection，statement/connection cleanup 集中化 |
| SQLite bind 没检查无 active statement | P2 | 增加状态检查和明确异常 |
| GroupManager completion callback 抛异常可终止聚合线程 | P1 | callback exception boundary，之后仍允许相机进入下一组 |
| `trtyolo_slice.hpp` 引入完整 OpenCV 和日志实现 | P2 | 收窄公共头依赖、移除日志头、明确 deep-copy 意图 |
| `SliceDetector(unique_ptr)` 可接受 null model | P2 | 构造时立即拒绝 null |
| Python API 工具与当前 Server 参数契约漂移 | P2 | 重写 `api_add.py`，删除无效历史字段 |
| Python 工具重复 config/HTTP 代码 | P2 | 新增 `api_common.py` |
| `api_healty.py` 文件名拼写错误 | P3 | 改为 `api_health.py` |

---

# 文件级审查

## `main.cpp`

当前职责清晰：CLI -> Config -> DB bootstrap -> run_id -> Server。

后续建议：

- `makeRunId()` 可移动到 Runtime/Utils，但目前留在入口更容易理解；
- 若未来支持多 run 动态切换，run_id 生命周期应进入专门 `RuntimeContext`，不要重新散回 Scheduler。

状态：**可接受**。

---

## `Core/Config.hpp` / `Core/Config.cpp`

优点：

- 相对路径以配置文件目录解析；
- 端口、线程数、timeout、GPU id 有基本校验；
- 目录创建错误会使配置无效。

问题：

1. **P2：Config 是 public mutable data bag。** 任意模块可以运行中修改 `outputdir`、端口等。长期可改为 private + accessors 或 immutable value object。
2. **P2：`mState` 命名不表达语义。** 推荐最终替换为 `isValid()`；考虑兼容现有代码，本轮不做大面积 rename。
3. **P2：Config 构造同时读取、校验、路径归一化并创建目录。** 如果部署逻辑继续增长，建议拆为 `ConfigLoader` + `Config::validate()`。

状态：**本轮不改变 public API，避免无编译环境下扩大修改面**。

---

## `Core/runtime_state.hpp`

职责说明正确，但当前仍有两个消费者违反其注释中的目标：

- `DetectThread` 读取 `g_runtime_state.config/run_id`；
- `SingleDetectThread` 读取 `g_runtime_state.config/run_id`。

**P1/P2：Service Locator 式全局读取。** Worker 应接收明确的 output context，例如：

```cpp
struct DetectionContext {
    std::filesystem::path output_root;
    std::string run_id;
};
```

之后 Scheduler/Server 构造 Worker 时传入。

状态：**待下一轮依赖注入重构**。

---

## `Core/detect_params.hpp`

已从 `Server.hpp` 解耦，是正确方向。

问题：

- **P2：一个 struct 聚合预处理、模型、测量、相机身份、执行设备多个子域。** 长期可拆为 `PreprocessParams / DetectionParams / MeasurementParams / ExecutionParams`；
- 默认 `allowed_class_indices{0,2}` 是业务语义，虽然已可由 API 覆盖，但最好进入部署配置或模型 profile，而非通用类型默认值。

状态：**结构清晰，可继续分组**。

---

## `Core/Server.hpp`

公共头已经缩到非常小，未再暴露 libevent/WinSock 和 DetectParams。

状态：**良好**。

---

## `Core/Server.cpp`

当前仍是最大的控制面实现之一，包含：

- libevent transport；
- query/JSON 参数解析；
- HTTP request -> DetectParams mapping；
- 参数校验；
- model path resolution；
- response serialization；
- Scheduler/SingleDetect/Camera endpoint 生命周期。

问题：

1. **P1/P2：职责仍偏多。** 推荐后续拆 `RequestParams`、request mapper、response serializer。
2. **P1：`/api/single_detect/add` 同步等待 GPU/HALCON 完成，运行在 libevent 回调线程。** 一个长任务会阻塞其它 HTTP 请求。长期需要 job API 或后台 future。
3. **P1：单图正在执行时 `stop()` 无法真正打断 GPU/HALCON in-flight operation。** 当前只能取消队列中的 pending task。
4. 所有业务错误仍通过 HTTP 200 + body code 表达。这是现有协议约定，不在静态重构中改变。

状态：**控制面已明显改善，但仍建议继续拆**。

---

## `Core/Scheduler.hpp` / `Core/Scheduler.cpp`

本轮移除了 SQL、文件路径回填和 domain persistence 细节。

现在职责：

- Queue 生命周期；
- Camera 生命周期；
- N 个 DetectThread；
- GroupManager；
- 将 group-complete event 连接到 Repository。

仍存在：

- **P1：DetectThread::start() 没有 readiness/failure handshake。** 模型可能在线程内部初始化失败，但 Scheduler 已返回 started。建议未来提供 `startAndWaitReady()` 或状态 enum。

状态：**职责边界明显改善**。

---

## `Core/queue_manager.hpp` / `Core/queue_manager.cpp`

原实现先 `tryPop()`，随后才进入无谓词 `condition_variable::wait()`。生产者可能刚好在两者之间 enqueue + notify，造成消费者错过通知。

本轮已改为：

- item count；
- `wait(lock, predicate)` / `wait_until(lock, predicate)`；
- stop 使用 acquire/release 可见性；
- size API 返回内部 count，不再依赖 `size_approx()`。

剩余问题：

- **P1：队列仍无容量上限/backpressure。** 推理持续落后于采集时可能增长内存。工业服务建议增加 max raw/result queue 和明确 overflow policy。

状态：**并发语义修复，容量策略待设计**。

---

## `Core/camera_thread.hpp` / `Core/camera_thread.cpp`

职责：相机资源 + capture loop + slide protocol orchestration。

优点：

- 单图 timeout 已真正传到底层 Snap；
- stop 会终止 pending single-capture promise；
- CameraThread 不再负责 QueueManager start/stop。

风险：

1. **P0/P1：group capture 的 Snap/通道异常仍无限重试当前 face。** 生产相机暂时异常时线程可能永不完成当前组。代码中已明确保留历史语义，需实机定义 retry budget。
2. **P2：slide protocol 继续增长时 CameraThread 会再次膨胀。** 推荐 `SlideController` 封装 LOAD/GOTO/CAM_NEXT/UNLOAD 状态机。
3. 当前逻辑始终打开 SDK 枚举的第 0 台相机，`device_id` 是业务标签而不是物理设备选择器。

状态：**不静态改变硬件行为**。

---

## `Core/slide_serial.hpp` / `Core/slide_serial.cpp`

职责相对纯粹：Windows serial transport + line reader buffer。

风险：

- `Close()` 等 reader thread 退出依赖同步 `ReadFile` timeout 返回；timeout 配置过大时 stop latency 会增大；
- COM1..64 探测是 best-effort，不应当被上层当作完整设备枚举。

状态：**可接受**。

---

## `Core/group_manager.hpp` / `Core/group_manager.cpp`

职责正确：只聚合四面结果和处理 timeout。

本轮修复 callback exception boundary。

剩余问题：

- **P1/P2：stop 时尚未完成的 mGroups 会直接丢弃。** 需要业务决定 cancel 时是否应该持久化 ABORTED/TIMEOUT；
- completion callback 是同步执行，数据库写入时间会影响允许下一组的时机。当前这能形成天然节流，但应明确这是设计还是偶然行为。

状态：**核心逻辑清晰**。

---

## `Core/image_types.hpp`

DTO 已有 `processing_ok/error_message`，成功无检测和算法失败可以区分，这是重要改进。

问题：

- `QuadFrameResult::source` 当前 GroupManager 没有填充，实际 group id 从 results 获取；字段语义需要决定保留还是删除；
- 若结果状态继续扩大，建议把 bool + string 替换为 enum `ProcessingStatus` + diagnostic。

状态：**可接受**。

---

## `Core/inspection_repository.hpp` / `Core/inspection_repository.cpp`

本轮新增。

职责：

- group/face/detection table mapping；
- transaction；
- group status persistence；
- 缺省 original path 回填。

所有 face_id lookup 已使用 prepared SQL，无运行时字符串拼接。

后续：如果需要查询 API，再拆 read repository；不要把 SELECT API 堆回 Scheduler。

状态：**新增职责边界**。

---

## `Core/sqlite_helper.hpp` / `Core/sqlite_helper.cpp`

本轮改进：

- constructor failure cleanup；
- busy timeout 返回码检查；
- statement cleanup 集中；
- bind 前检查 active statement；
- 增加 `stepInt64()` 支持参数化 scalar SELECT。

问题：

- `query(string)` 仍是 raw SQL API；新业务代码应避免拼接外部数据；
- 若 SQL 使用继续增加，建议把 helper 做成真正 `Statement` RAII object，而不是 connection 内只允许一个 `stmt_`。

状态：**足够支撑当前 repository**。

---

## `Core/db_utils.hpp` / `Core/db_utils.cpp`

职责应该叫 schema bootstrap，而不是 migration。

**P1/P2：当前 `CREATE TABLE IF NOT EXISTS` 不会迁移旧 schema。** 如果线上已有历史数据库，新版本字段变化不能依靠这个函数完成迁移。

推荐：

```sql
PRAGMA user_version;
-- v1 -> v2 migration
-- v2 -> v3 migration
```

状态：**必须在下次 schema 变化前建立 migration 机制**。

---

## `Core/detect_thread.hpp` / `Core/detect_thread.cpp`

优点：

- thread-local CUDA/model 生命周期；
- GPU device 来自参数；
- HALCON/切片失败会发布 failed result，不再默默丢帧；
- centers empty 被明确视为有效无缺陷结果。

问题：

1. **P1/P2：读取 `g_runtime_state` 获取输出路径。** 应显式注入 runtime output context。
2. **P1/P2：与 SingleDetectThread 重复一整套 pipeline。** 长期抽 `DetectionPipeline`。
3. Worker 模型初始化失败只把自己的 `mRunning=false`，Scheduler 无 readiness 感知。

状态：**处理错误能力改善，架构去重待做**。

---

## `Core/single_detect_thread.hpp` / `Core/single_detect_thread.cpp`

优点：任务队列、缓存模型、GPU 切换时销毁 device-bound resource 均较清楚。

问题：

1. **P1：submitAndWait 无 deadline。** 算法调用卡住会永久阻塞 HTTP callback；
2. **P1/P2：与 DetectThread 的 preprocess/HALCON/slice/YOLO/SAM 重复；**
3. **P2：读取全局 RuntimeState 获取 output path；**
4. active task 无法被 stop 强制中断。

状态：**建议下一阶段重点处理**。

---

# `Core/detect/` 文件级审查

## `crop_image.h` / `crop_image.cpp`

函数名已经不能描述真实职责：它不仅 crop，还执行 stripe removal、denoise、QW mask、保存；同时拥有 20+ 参数。

**P1/P2：parameter explosion。** 推荐兼容式演进：

```cpp
struct ImagePreprocessOptions {
    CropOptions crop;
    QwMaskOptions qw;
    StripeOptions stripe;
    SaveOptions save;
};

GpuMat preprocessImage(input, options, stream);
```

先增加新 API 并迁移内部调用，最后保留/弃用旧 `cropImage` wrapper。由于参数顺序直接影响算法行为，本轮没有在无编译/数据验证环境下强改。

---

## `StripeRemoval.h` / `StripeRemoval.cpp`

优点：当前 class 无共享 mutable state，适合多 worker。

问题：

- 单个 `remove_image_stripes()` 仍很长，可按 DFT、mask generation、inverse transform、denoise 拆 private helper；
- CPU 方向检测要求显式 stream sync，当前 caller 已同步但 contract 应更明显；
- 输入允许 1/3/4 channel，但输出固定 BGR 3-channel，需要在 API 注释中明确；
- 使用 OpenMP pragma，项目配置是否统一启用 OpenMP 需要构建机验证。

状态：**不改变数学行为**。

---

## `HalconProcessor.h` / `HalconProcessor.cpp`

异步保存队列与 callback 边界整体合理。

问题：

1. **P2：每个 HalconProcessor 构造都会调用全局 `HalconCpp::SetSystem`。** 推荐 `std::call_once` 做进程级 HALCON 初始化；
2. 日志混用 `std::cerr` 与项目 `LOGE/LOGI`；
3. async save 返回 future，但多数 caller 不消费，真正同步由 `waitForAsyncOperations()` 完成；API 可以简化或明确两种模式。

状态：**建议低风险后续清理**。

---

## `Slice.h` / `Slice.cpp`

CPU/GPU 切片的边界填零逻辑清晰，GPU 返回前同步下载，避免 CPU Mat 未完成的问题。

问题：

- 全局 `using SliceInfo = trtyolo::SliceInfo` 污染 namespace；
- Server 已拒绝负 `slice_distance`，这里又把负数做 `abs`，contract 不一致；
- CPU/GPU 两条循环有重复但不严重。

状态：**可接受**。

---

## `trtyolo_slice.hpp` / `trtyolo_slice.cpp`

本轮：

- 缩小公共 header include；
- 日志 include 移到 cpp；
- SliceInfo 明确 deep copy；
- null model fail-fast；
- threshold 参数增加防御式校验。

需实模验证：

- 当前 NMS 是 class-agnostic；
- 恢复到原图的 box 只 clamp 下界，没有源图 width/height，因此边缘 box 可能超出右/下边界。

状态：**行为保持，边界策略待指标验证**。

---

## `sam.h` / `sam.cpp`

**接口副作用需要更明显：** `inferFromDetections()` 会原地过滤并重写传入的 `DetectRes&`，不仅“infer”。建议未来改名为 `segmentAndFilterDetections` 或返回新结果结构。

面积/直径缓存与过滤后的 detection 一一对应，这一点当前实现是合理的。

状态：**命名/API 可进一步优化**。

---

## `speedSam.h` / `speedSam.cpp`

两个重要问题需要真实模型验证：

1. **P1 性能：同一张图对每个 detection 都调用一次 `SpeedSam::predict()`，每次重新跑 image encoder。** SAM encoder 应该按 image 一次，decoder 按 prompt N 次或 batch；
2. **P0/P1 模型语义：decoder 输出有 `NUM_LABELS` 个 mask 与 IoU prediction，但当前代码用 `mLowResMasks.data()` 的第一个 mask 构造结果，没有按 IoU 选择最佳 mask。** 必须结合当前导出的 SAM decoder 语义确认，不能静态猜测哪一 mask 正确。

小问题：存在不必要的 `const_cast<float*>(labels.data())`，因为 EngineTRT 参数本身接受 `const float*`。

状态：**优先安排模型级回归实验**。

---

## `engineTRT.h` / `engineTRT.cpp`

风险：

1. **P1：class 成员仍是 raw TensorRT/CUDA resources。** constructor 中后半段抛异常时，完整对象 destructor 不会运行，存在部分构造资源泄漏可能。推荐统一 `release() noexcept` 并在 constructor catch 中清理，或用 custom-deleter unique_ptr；
2. **P0/P1：代码虽然接收 input/output tensor names，但 `initialize()` 实际忽略 names，decoder `setInput()` 假设 tensor index 0..4 的固定顺序；** 不同 engine 的 IO tensor enumeration 顺序不应被隐式当作业务契约；
3. `getOutput(iou, masks)` 用 buffer byte size 判断哪个 output 是 IoU、哪个是 mask，属于脆弱启发式；应按 tensor name 显式映射；
4. dynamic prompt profile max=10，而 `config.h` 中 `MAX_NUM_PROMPTS=1`，需要梳理这个常量究竟只用于静态 shape fallback 还是实际限制。

状态：**下一轮高优先级底层重构，但必须结合当前 TensorRT 10 engine 验证**。

---

## `config.h`

大量通用宏：`MODEL_INPUT_WIDTH`、`HIDDEN_DIM` 等。

**P2：宏污染全局命名空间。** 最终应改为：

```cpp
namespace sam_config {
inline constexpr int kInputWidth = 1024;
}
```

由于 engine/SAM 代码广泛引用，本轮记录而不做纯风格大改。

---

## `logging.h`

明显来自 TensorRT sample 风格 logger，代码很长但属于相对独立的第三方适配层。

建议：

- 标注来源/许可证（如果来自 NVIDIA sample）；
- 不把业务日志继续写进此文件；
- EngineTRT 之外的项目代码统一使用 `Core/Utils/Log.hpp`。

状态：**视作基础适配文件，不为“统一风格”强改**。

---

## `macros.h`

仅提供 DLL/TensorRT 兼容宏，职责明确。

状态：**保留**。

---

## `cuda_utils.h`

宏依赖 `std::cerr`、`assert`、`sample::gLogError`、`FN_NAME` 等外部符号，但自身没有完整声明依赖。

**P2：不是自包含 header。** 如果已无调用，建议删除；若仍有第三方代码依赖，应补齐来源和依赖。不要继续在新代码中使用 assert 型 CUDA error handling。

状态：**待确认外部依赖后删除或隔离**。

---

# 工程与工具

## `mt.vcxproj`

已有进步：ThirdParty/TensorRT 不再写个人绝对盘符，改用 MSBuild properties。

仍有：

- **P1/P2：只有 Release|x64 配置了完整 include/link dependencies。** Debug x64 / Win32 目标事实上不是同等可构建配置；
- CUDA 版本仍固定为 11.8 build customization；应和实际 OpenCV CUDA、TensorRT 组合一起锁定；
- 第三方目录名 `opencv[cuda118]` 带特殊字符，可用但维护体验一般。

状态：**需要在 Windows 构建机上整理成统一 props**。

---

## `mt.vcxproj.filters`

仅是 IDE 视图。本轮已经加入 `InspectionRepository`。

状态：**同步工程文件即可**。

---

## `mt.sln`

单项目 solution，没有业务逻辑。

状态：**无需静态重构**。

---

## `.gitignore`

覆盖 Visual Studio 构建产物、output、env 等。

状态：**基本合理**。

---

## `.vscode/settings.json`

只有文件类型关联，没有机器绝对路径。

状态：**低风险，可保留；如果团队不统一使用 VS Code 也可以删除**。

---

## `pytesttool/api_common.py`

本轮新增，共享 endpoint/config/JSON HTTP 调用。

---

## `pytesttool/api_add.py`

本轮改为只发送当前 Server 实际解析的字段，并使用 kebab-case CLI -> JSON snake_case 映射。

---

## `pytesttool/api_cancel.py`

本轮移除重复的配置读取和 urllib 错误处理。

---

## `pytesttool/api_health.py`

由历史拼写错误 `api_healty.py` 更名，复用公共 client。

---

# 下一阶段建议顺序

1. **先做可测试的代码去重**：抽 `DetectionPipeline`，让单图/多图调用同一核心处理函数；
2. **消除 Worker 对 RuntimeState 的读取**：显式 `DetectionContext`；
3. **TensorRT IO name mapping + constructor RAII**；
4. **建立 worker READY/FAILED handshake**；
5. **Queue backpressure + camera retry budget**；
6. **Server request mapper / HTTP transport 拆分**；
7. **SQLite `PRAGMA user_version` migration**；
8. 最后再做命名、namespace、宏常量等广泛风格统一。

这个顺序优先减少真实故障面，再进行美观型重构。
