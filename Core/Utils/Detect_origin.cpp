int main() {
   try {
       std::string uuid = "001";
       std::string savePath = "./output";
       // 初始化
       HalconProcessor processor;
       std::string trt_engine_file = "E:\\ydk\\resource\\models\\v12s3-20.engine";
       trtyolo::InferOption infer_option;
       infer_option.enableSwapRB();  // 启用RGB通道交换
       trtyolo::SliceDetector slice_detector(trt_engine_file, infer_option);
       SamSegmenter seg;
       bool sam_ok = seg.init(
           "E:\\ydk\\resource\\models\\SAM\\SAM_encoder.engine",
           "E:\\ydk\\resource\\models\\SAM\\SAM_mask_decoder.engine"
       );
       if (!sam_ok) {
           std::cerr << "错误: SAM 初始化失败!" << std::endl;
           return -1;
       }

       auto total_start = std::chrono::high_resolution_clock::now();
       auto stage_start = total_start;
       int x = 0, y = 0, width = 2856, height = 2300;
       int x1 = 314, y1 = 1583, x2 = 2392, y2 = 1588, radius = 316 - 130;
       // 1. 读取原始图像
       cv::Mat originalImage = cv::imread("E:\\ydk\\op\\mt\\output\\111.png");
       if (originalImage.empty()) {
           std::cout << "无法加载图像文件！" << std::endl;
           return -1;
       }
       cv::cuda::GpuMat d_input;
       d_input.upload(originalImage);
       cv::cuda::GpuMat d_qwCropped = cropImage(d_input,
           true,
           x,
           y,
           width,
           height,
           true,
           x1,
           y1,
           x2,
           y2,
           radius,
           "",
           "",
           true,
           10,     // filter_width
           0.0001, // attenuation_factor
           90,     // target_angle
           10,     // angle_tolerance
           true,   // enable_denoising
           8.0f,   // denoise_h
           8.0f,   // denoise_hColor
           17,     // denoise_searchWindowSize
           11      // denoise_templateWindowSize
       );
       cv::cuda::GpuMat d_qwCropped1 = cropImage(d_input, true, x, y, width, height, true, x1, y1, x2, y2, radius);
       cv::Mat qwCropped1;
       d_qwCropped1.download(qwCropped1);
       auto stage_end = std::chrono::high_resolution_clock::now();
       auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stage_end - stage_start);
       std::cout << "裁剪和条纹去除完成，耗时: " << duration.count() << "ms" << std::endl;

       // 2. 使用HalconProcessor处理图像获取中心点（直接传入GPU图像）
       stage_start = std::chrono::high_resolution_clock::now();
       std::string outputImage;
       std::vector<std::pair<double, double>> centerPoints;


       bool result = processor.processImage(d_qwCropped, outputImage, centerPoints, true, uuid, savePath);
       if (!result) {
           std::cerr << "错误: HalconProcessor处理失败!" << std::endl;
           return -1;
       }

       stage_end = std::chrono::high_resolution_clock::now();
       duration = std::chrono::duration_cast<std::chrono::milliseconds>(stage_end - stage_start);
       std::cout << "成功获取 " << centerPoints.size() << " 个中心点，耗时: " << duration.count() << "ms" << std::endl;

       // 3. 提取切片
       stage_start = std::chrono::high_resolution_clock::now();
       std::vector<SliceInfo> slices = ImageSlicer::extractSlicesGPU(
           d_qwCropped,
           centerPoints,
           40,                           // 距离参数
           false,                        // 启用保存
           "./slices",                   // 保存路径
           640,                          // 切片宽度
           640                           // 切片高度
       );

       if (slices.empty()) {
           std::cerr << "错误: 未提取到任何切片!" << std::endl;
           return -1;
       }
       stage_end = std::chrono::high_resolution_clock::now();
       duration = std::chrono::duration_cast<std::chrono::milliseconds>(stage_end - stage_start);
       std::cout << "成功提取 " << slices.size() << " 个切片，耗时: " << duration.count() << "ms" << std::endl;


       // 5. 直接使用提取的切片
       stage_start = std::chrono::high_resolution_clock::now();
       const auto& trt_slices = slices;
       stage_end = std::chrono::high_resolution_clock::now();
       duration = std::chrono::duration_cast<std::chrono::milliseconds>(stage_end - stage_start);
       std::cout << "切片准备完成，耗时: " << duration.count() << "ms" << std::endl;

       // 6. 进行切片推理
       stage_start = std::chrono::high_resolution_clock::now();
       // 仅保留需要的类别（示例：索引为 0 和 2 的类别）
       std::vector<int> allowed_classes = { 0, 2 };
       trtyolo::DetectRes detection_result = slice_detector.process_sliced_images(
           trt_slices,
           0.45f,    // NMS阈值
           0.2f,     // 置信度阈值
           allowed_classes  // 过滤的类别索引
       );
       stage_end = std::chrono::high_resolution_clock::now();
       duration = std::chrono::duration_cast<std::chrono::milliseconds>(stage_end - stage_start);
       std::cout << "推理完成，耗时: " << duration.count() << "ms" << std::endl;
       std::cout << "检测到 " << detection_result.num << " 个目标" << std::endl;

       // 计算推理速度
       double inference_time_seconds = duration.count() / 1000.0;
       double fps = slices.size() / inference_time_seconds;
       std::cout << "推理速度: " << fps << " FPS (" << inference_time_seconds << " 秒处理 " << slices.size() << " 个切片)" << std::endl;

       // 7. 使用 SAM 进行分割并可视化
       stage_start = std::chrono::high_resolution_clock::now();

       // 使用检测框作为提示进行分割推理
       auto masks = seg.inferFromDetections(qwCropped1, detection_result, 100, 10);

       // 可视化（透明红色覆盖 + 绘制检测框），并保存到 savePath/uuid/seg_<uuid>.png
       bool enableSegSave = true;

       cv::Mat seg_vis = seg.visualize(d_qwCropped1, detection_result, masks, enableSegSave, uuid, savePath, /*image_index*/ 0);

       // 额外保存一份检测可视化（可选）
       // 定义类别标签（根据你的模型调整）
       std::vector<std::string> labels = {
           "hh-Y", "class_1", "ox-Y", "class_3", "class_4",
           "class_5", "class_6"
       };

       stage_end = std::chrono::high_resolution_clock::now();
       duration = std::chrono::duration_cast<std::chrono::milliseconds>(stage_end - stage_start);
       std::cout << "SAM 分割可视化和保存完成，耗时: " << duration.count() << "ms" << std::endl;

       // 8. 打印详细的检测结果
       std::cout << "\n=== 检测结果详情 ===" << std::endl;
       for (int i = 0; i < detection_result.num; ++i) {
           const auto& box = detection_result.boxes[i];
           int class_id = detection_result.classes[i];
           float score = detection_result.scores[i];

           std::string class_name = (class_id < labels.size()) ? labels[class_id] : "unknown";

           std::cout << "目标 " << i + 1 << ": "
               << "类别=" << class_name
               << "(" << class_id << "), "
               << "置信度=" << score
               << ", 位置=(" << box.left << "," << box.top
               << "," << box.right << "," << box.bottom << ")" << std::endl;
       }

       // 总耗时统计
       auto total_end = std::chrono::high_resolution_clock::now();
       auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
       std::cout << "\n=== 总耗时统计 ===" << std::endl;
       std::cout << "程序总执行时间: " << total_duration.count() << "ms ("
           << total_duration.count() / 1000.0 << "秒)" << std::endl;

       std::cout << "\n程序执行成功!" << std::endl;
       return 0;

   }
   catch (const std::exception& e) {
       std::cerr << "程序执行出错: " << e.what() << std::endl;
       return -1;
   }
   catch (...) {
       std::cerr << "程序执行出错: 未知异常" << std::endl;
       return -1;
   }
}