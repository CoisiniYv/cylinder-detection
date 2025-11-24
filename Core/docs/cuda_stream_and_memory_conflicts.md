# CUDA 流与显存访问冲突梳理（基于 detect_thread.cpp#105-309）

## 概览
- 本线程在 `run()` 中创建了线程专属 `cv::cuda::Stream`（`detect_thread.cpp:49`）。
- 仅以下操作明确使用了该非默认流：`GpuMat::upload(src, stream)` 与 `GpuMat::download(vis_cpu, stream)`，以及随后一次 `stream.waitForCompletion()`（`detect_thread.cpp:132, 265-266`）。
- 绝大多数 GPU 算子（裁剪、频域处理、切片、去噪、merge/multiply/convertTo、copyTo 等）均未传入 `Stream` 参数，因而落在 OpenCV 默认流（legacy default stream）上。
- 这形成了“自有流 + 默认流并存”的混合流水线，若无正确的跨流同步，会出现显存读写竞态以及可见的下载时机错误。

## 非默认流的使用点
- `cv::cuda::Stream stream;` 初始化：`detect_thread.cpp:49`
- `d_input.upload(src, stream)`：`detect_thread.cpp:132`
- `d_cropped.download(vis_cpu, stream)`：`detect_thread.cpp:265`
- `stream.waitForCompletion()`：`detect_thread.cpp:266`

## 默认流的使用点（未显式传入 Stream）
- 裁剪与频域处理路径：`cropImage(...)`（`detect/crop_image.cpp`），内部包含：
  - ROI `clone`、`copyTo`、`merge`、`multiply`、`convertTo`、`download` 等，均为默认流。
  - 条纹去除 `StripeRemoval::remove_image_stripes(...)`（`detect/StripeRemoval.cpp`）中的 `magnitude`、FFT相关拷贝、`fastNlMeansDenoisingColored`、`download` 等，均为默认流。
- 切片提取：`ImageSlicer::extractSlicesGPU(...)`（`detect/Slice.cpp`）中的 `setTo`、`copyTo`、`download`，默认流。
- HALCON 处理：`HalconProcessor::gpuMatToHalconHObject(...)`（`detect/HalconProcessor.cpp`）将 `GpuMat` 下载到 CPU，使用默认流。

## 关键竞态与冲突场景
1. 上传-计算跨流竞态（默认流读取了尚未完成的自有流上传）
   - 步骤：`upload(src, stream)` → 紧接着调用 `cropImage(d_input, ...)`。
   - 问题：`cropImage` 内的 GPU 算子在默认流上运行，不会自动等待 `stream` 上的异步上传完成；如果上传尚未落盘到设备内存，默认流的读操作可能读到未定义数据，造成偶发错误或结果随机。

2. 计算-下载跨流竞态（默认流写未完成，但在自有流上下载）
   - 步骤：默认流上完成 `cropImage`/`StripeRemoval`/`Slice` 的写入 → 使用 `d_cropped.download(vis_cpu, stream)` → `stream.waitForCompletion()`。
   - 问题：`waitForCompletion()` 仅等待 `stream` 队列；默认流的计算未被该等待覆盖。若默认流上的写入仍在进行，`stream` 上的下载可能复制到 CPU 的是未最终写好的显存，导致图像不完整或数据损坏。

3. 多线程默认流共享导致顺序不可控
   - 每个检测线程都有各自的 `cv::cuda::Stream`，但绝大多数算子仍落在同一个默认流。多个线程并发向默认流入队时，其执行顺序与阻塞关系受 CUDA 默认流语义影响，可能出现意外串行或跨线程写读交错，进一步放大显存竞态风险。

4. GPU→CPU 下载时机错配
   - 在 `cropImage`、`StripeRemoval`、`Slice`、`HalconProcessor::gpuMatToHalconHObject` 中存在多次 `download(...)` 到 CPU，均依赖默认流的完成时机。
   - 若调用方在自有流上进行 `download` 或等待自有流完成（而非默认流），则无法正确保证前置默认流写入已全部完成。

## 典型症状（与显存访问冲突一致）
- 偶发的图像下载失败或得到全零/随机数据。
- CUDA 报错如 `unspecified launch failure`、`invalid device pointer`、`illegal memory access`。
- 结果随机、跨线程干扰、偶发崩溃，随负载变化（线程数、图像大小）放大。

## 建议的同步与流使用策略（不改代码的思路说明）
- 流统一：将整条流水线统一到“同一个 `cv::cuda::Stream`”，避免默认流与自有流混用。若函数无流参数，需在函数内增加流支持或在调用前后做统一的同步屏障。
- 明确跨流同步：
  - 在调用默认流算子之前，确保自有流上的上传/前置操作已完成（如使用 `cudaEvent_t` 在自有流记录事件，并在默认流等待该事件）。
  - 在自有流上的下载之前，确保默认流上的写入全部完成（同样使用事件或统一使用默认流）。
- 统一下载流：避免“默认流写入 + 自有流下载”的组合；若无法改算子流，改为默认流下载并在默认流上 `waitForCompletion()`。
- 线程隔离：不同检测线程使用各自独立流时，尽量保证该线程内所有算子都在其流上执行，减少跨线程共享默认流。
- 调试屏障：在定位问题阶段可暂时使用 `cudaDeviceSynchronize()` 或 OpenCV 的同步下载（不传流）来验证竞态是否消失，从而确认冲突来源。

## 逐段梳理（对应代码路径）
- 上传到 GPU：`detect_thread.cpp:132`
  - 使用了自有流 `stream`（异步）。
  - 后续的 `cropImage` 立即在默认流上读取 `d_input`，存在上传未完成时的读竞态。

- 裁剪与条纹处理：`detect/crop_image.cpp`、`detect/StripeRemoval.cpp`
  - 所有 GPU 算子默认流执行；期间的 `download(...)` 也不带流参数。
  - 与自有流无显式同步，易出现写-读跨流时序问题。

- 切片提取：`detect/Slice.cpp`
  - 默认流执行，末尾 `sliceGPU.download(...)` 默认流下载。

- HALCON 转换：`detect/HalconProcessor.cpp`
  - `gpuMatToHalconHObject(...)` 中 `gpuMat.download(mat)` 默认流下载。
  - 若调用点在自有流上等待，无法覆盖默认流的未完成工作。

- 可视化下载与等待：`detect_thread.cpp:265-266`
  - 在自有流上 `download(vis_cpu, stream)` 并 `stream.waitForCompletion()`，但未等待默认流，存在跨流未完成风险。

## 落地方案建议（供后续实现参考）
- 为 `cropImage`、`StripeRemoval::remove_image_stripes`、`ImageSlicer::extractSlicesGPU`、`HalconProcessor::gpuMatToHalconHObject` 增加 `cv::cuda::Stream` 参数，并在内部将所有 GPU 算子切换到该流。
- 若短期无法改动，至少在关键边界插入正确的同步：
  - 在 `upload(src, stream)` 后立即 `stream.waitForCompletion()` 再进入默认流计算（验证竞态是否消失）。
  - 在默认流所有写入完成后，再进行自有流的下载；或统一在默认流上下载并等待默认流。
- 使用 CUDA 事件在默认流与自有流之间建立依赖关系，消除跨流间的隐式竞态。

## 快速检查清单
- 同一张 `GpuMat` 是否在“上传/写入”与“读取/下载”之间跨了不同流？
- 是否仅等待了自有流而未等待默认流？
- 是否在多线程环境共享了默认流上的算子，导致隐藏的顺序问题？
- HALCON 下载之前，GPU 写入是否已经在正确的流上完成？

以上为当前代码的流使用与显存冲突梳理，供后续修复与重构参考。