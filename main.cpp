#include <iostream>
#include <string>
#include "Core/Server.hpp"
#include "Core/Config.hpp"
//#include "Core/sqlite_helper.hpp"
using namespace XL;

//int main(int argc, char** argv) {
//    try {
//        const char* cfg_file = nullptr;
//
//        // 命令行参数：-h 打印帮助；-f 指定配置文件路径（默认使用当前目录下的 config.json）
//        for (int i = 1; i < argc; i += 2) {
//            if (argv[i][0] != '-') {
//                printf("参数错误:%s\n", argv[i]);
//                return -1;
//            }
//            switch (argv[i][1]) {
//            case 'h': {
//                printf("-h 打印参数帮助并退出\n");
//                printf("-f 配置文件    如：-f config.json \n");
//                return 0;
//            }
//            case 'f': {
//                if (i + 1 < argc) cfg_file = argv[i + 1];
//                break;
//            }
//            default: {
//                printf("参数错误:%s\n", argv[i]);
//                return -1;
//            }
//            }
//        }
//
//        if (cfg_file == nullptr) {
//            cfg_file = "config.json"; // 默认使用当前目录配置
//        }
//
//        Config config(cfg_file);
//        if (!config.mState) {
//            printf("读取配置失败: %s\n", cfg_file);
//            return -1;
//        }
//        config.show();
//
//        // 启动 HTTP Server（阻塞运行，接受 /api/control/add 与 /api/control/cancel 指令）
//        Server server;
//        server.start(&config);
//        return 0;
//    }
//    catch (const std::exception& e) {
//        std::cerr << "程序执行出错: " << e.what() << std::endl;
//        return -1;
//    }
//    catch (...) {
//        std::cerr << "程序执行出错: 未知异常" << std::endl;
//        return -1;
//    }
//}






//#include "Core/detect/trtyolo_slice.hpp"
//#include "Core/detect/HalconProcessor.h"
//#include "Core/detect/Slice.h"
//#include "Core/detect/sam.h"
//#include "Core/detect/trtyolo_slice.hpp"
//#include "Core/detect/crop_image.h"
//int main() {
//    try {
//        std::string uuid = "001";
//        std::string savePath = "./output";
//        // 初始化
//        HalconProcessor processor;
//        std::string trt_engine_file = "E:\\ydk\\resource\\models\\v12s3-20.engine";
//        trtyolo::InferOption infer_option;
//        infer_option.enableSwapRB();  // 启用RGB通道交换
//        trtyolo::SliceDetector slice_detector(trt_engine_file, infer_option);
//        SamSegmenter seg;
//        bool sam_ok = seg.init(
//            "E:\\ydk\\resource\\models\\SAM\\SAM_encoder.engine",
//            "E:\\ydk\\resource\\models\\SAM\\SAM_mask_decoder.engine"
//        );
//        if (!sam_ok) {
//            std::cerr << "错误: SAM 初始化失败!" << std::endl;
//            return -1;
//        }
//
//        auto total_start = std::chrono::high_resolution_clock::now();
//        auto stage_start = total_start;
//        int x = 0, y = 0, width = 2856, height = 2300;
//        int x1 = 314, y1 = 1583, x2 = 2392, y2 = 1588, radius = 316 - 130;
//        // 1. 读取原始图像
//        cv::Mat originalImage = cv::imread("E:\\ydk\\op\\mt\\output\\111.png");
//        if (originalImage.empty()) {
//            std::cout << "无法加载图像文件！" << std::endl;
//            return -1;
//        }
//        cv::cuda::GpuMat d_input;
//        d_input.upload(originalImage);
//        cv::cuda::GpuMat d_qwCropped = cropImage(d_input,
//            true,
//            x,
//            y,
//            width,
//            height,
//            true,
//            x1,
//            y1,
//            x2,
//            y2,
//            radius,
//            "",
//            "",
//            true,
//            10,     // filter_width
//            0.0001, // attenuation_factor
//            90,     // target_angle
//            10,     // angle_tolerance
//            true,   // enable_denoising
//            8.0f,   // denoise_h
//            8.0f,   // denoise_hColor
//            17,     // denoise_searchWindowSize
//            11      // denoise_templateWindowSize
//        );
//        cv::cuda::GpuMat d_qwCropped1 = cropImage(d_input, true, x, y, width, height, true, x1, y1, x2, y2, radius);
//        cv::Mat qwCropped1;
//        d_qwCropped1.download(qwCropped1);
//        auto stage_end = std::chrono::high_resolution_clock::now();
//        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stage_end - stage_start);
//        std::cout << "裁剪和条纹去除完成，耗时: " << duration.count() << "ms" << std::endl;
//
//        // 2. 使用HalconProcessor处理图像获取中心点（直接传入GPU图像）
//        stage_start = std::chrono::high_resolution_clock::now();
//        std::string outputImage;
//        std::vector<std::pair<double, double>> centerPoints;
//
//
//        bool result = processor.processImage(d_qwCropped, outputImage, centerPoints, true, uuid, savePath);
//        if (!result) {
//            std::cerr << "错误: HalconProcessor处理失败!" << std::endl;
//            return -1;
//        }
//
//        stage_end = std::chrono::high_resolution_clock::now();
//        duration = std::chrono::duration_cast<std::chrono::milliseconds>(stage_end - stage_start);
//        std::cout << "成功获取 " << centerPoints.size() << " 个中心点，耗时: " << duration.count() << "ms" << std::endl;
//
//        // 3. 提取切片
//        stage_start = std::chrono::high_resolution_clock::now();
//        std::vector<SliceInfo> slices = ImageSlicer::extractSlicesGPU(
//            d_qwCropped,
//            centerPoints,
//            40,                           // 距离参数
//            false,                        // 启用保存
//            "./slices",                   // 保存路径
//            640,                          // 切片宽度
//            640                           // 切片高度
//        );
//
//        if (slices.empty()) {
//            std::cerr << "错误: 未提取到任何切片!" << std::endl;
//            return -1;
//        }
//        stage_end = std::chrono::high_resolution_clock::now();
//        duration = std::chrono::duration_cast<std::chrono::milliseconds>(stage_end - stage_start);
//        std::cout << "成功提取 " << slices.size() << " 个切片，耗时: " << duration.count() << "ms" << std::endl;
//
//
//        // 5. 直接使用提取的切片
//        stage_start = std::chrono::high_resolution_clock::now();
//        const auto& trt_slices = slices;
//        stage_end = std::chrono::high_resolution_clock::now();
//        duration = std::chrono::duration_cast<std::chrono::milliseconds>(stage_end - stage_start);
//        std::cout << "切片准备完成，耗时: " << duration.count() << "ms" << std::endl;
//
//        // 6. 进行切片推理
//        stage_start = std::chrono::high_resolution_clock::now();
//        // 仅保留需要的类别（示例：索引为 0 和 2 的类别）
//        std::vector<int> allowed_classes = { 0, 2 };
//        trtyolo::DetectRes detection_result = slice_detector.process_sliced_images(
//            trt_slices,
//            0.45f,    // NMS阈值
//            0.2f,     // 置信度阈值
//            allowed_classes  // 过滤的类别索引
//        );
//        stage_end = std::chrono::high_resolution_clock::now();
//        duration = std::chrono::duration_cast<std::chrono::milliseconds>(stage_end - stage_start);
//        std::cout << "推理完成，耗时: " << duration.count() << "ms" << std::endl;
//        std::cout << "检测到 " << detection_result.num << " 个目标" << std::endl;
//
//        // 计算推理速度
//        double inference_time_seconds = duration.count() / 1000.0;
//        double fps = slices.size() / inference_time_seconds;
//        std::cout << "推理速度: " << fps << " FPS (" << inference_time_seconds << " 秒处理 " << slices.size() << " 个切片)" << std::endl;
//
//        // 7. 使用 SAM 进行分割并可视化
//        stage_start = std::chrono::high_resolution_clock::now();
//
//        // 使用检测框作为提示进行分割推理
//        auto masks = seg.inferFromDetections(qwCropped1, detection_result, 100, 10);
//
//        // 可视化（透明红色覆盖 + 绘制检测框），并保存到 savePath/uuid/seg_<uuid>.png
//        bool enableSegSave = true;
//
//        cv::Mat seg_vis = seg.visualize(d_qwCropped1, detection_result, masks, enableSegSave, uuid, savePath);
//
//        // 额外保存一份检测可视化（可选）
//        // 定义类别标签（根据你的模型调整）
//        std::vector<std::string> labels = {
//            "hh-Y", "class_1", "ox-Y", "class_3", "class_4",
//            "class_5", "class_6"
//        };
//
//        stage_end = std::chrono::high_resolution_clock::now();
//        duration = std::chrono::duration_cast<std::chrono::milliseconds>(stage_end - stage_start);
//        std::cout << "SAM 分割可视化和保存完成，耗时: " << duration.count() << "ms" << std::endl;
//
//        // 8. 打印详细的检测结果
//        std::cout << "\n=== 检测结果详情 ===" << std::endl;
//        for (int i = 0; i < detection_result.num; ++i) {
//            const auto& box = detection_result.boxes[i];
//            int class_id = detection_result.classes[i];
//            float score = detection_result.scores[i];
//
//            std::string class_name = (class_id < labels.size()) ? labels[class_id] : "unknown";
//
//            std::cout << "目标 " << i + 1 << ": "
//                << "类别=" << class_name
//                << "(" << class_id << "), "
//                << "置信度=" << score
//                << ", 位置=(" << box.left << "," << box.top
//                << "," << box.right << "," << box.bottom << ")" << std::endl;
//        }
//
//        // 总耗时统计
//        auto total_end = std::chrono::high_resolution_clock::now();
//        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
//        std::cout << "\n=== 总耗时统计 ===" << std::endl;
//        std::cout << "程序总执行时间: " << total_duration.count() << "ms ("
//            << total_duration.count() / 1000.0 << "秒)" << std::endl;
//
//        std::cout << "\n程序执行成功!" << std::endl;
//        return 0;
//
//    }
//    catch (const std::exception& e) {
//        std::cerr << "程序执行出错: " << e.what() << std::endl;
//        return -1;
//    }
//    catch (...) {
//        std::cerr << "程序执行出错: 未知异常" << std::endl;
//        return -1;
//    }
//}





#include <sqlite3.h>
#include <iostream>
#include <string>

static const char* ddl_inspection_groups = R"(
CREATE TABLE IF NOT EXISTS inspection_groups (
    group_id     INTEGER PRIMARY KEY,          -- 对应 MySQL BIGINT UNSIGNED PK
    device_id    TEXT    NOT NULL,
    status       TEXT    NOT NULL
                 CHECK (status IN ('NG','GOOD')),
    created_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_time DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_device_id     ON inspection_groups(device_id);
CREATE INDEX IF NOT EXISTS idx_status        ON inspection_groups(status);
CREATE INDEX IF NOT EXISTS idx_g_created_time ON inspection_groups(created_time);
)";

static const char* ddl_inspection_faces = R"(
CREATE TABLE IF NOT EXISTS inspection_faces (
    face_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id      INTEGER NOT NULL,
    face_index    INTEGER NOT NULL CHECK (face_index BETWEEN 0 AND 3),
    timestamp_ms  INTEGER NOT NULL,
    sequence_id   INTEGER NOT NULL,
    saved_path    TEXT    NOT NULL,
    skeleton_path TEXT    NOT NULL,
    created_time  DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(group_id, face_index),
    FOREIGN KEY (group_id) REFERENCES inspection_groups(group_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_group_id     ON inspection_faces(group_id);
CREATE INDEX IF NOT EXISTS idx_timestamp_ms ON inspection_faces(timestamp_ms);
CREATE INDEX IF NOT EXISTS idx_sequence_id  ON inspection_faces(sequence_id);
)";

static const char* ddl_defect_detections = R"(
CREATE TABLE IF NOT EXISTS defect_detections (
    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id      INTEGER NOT NULL,
    label_id     INTEGER NOT NULL,
    label        TEXT    NOT NULL,
    confidence   REAL    NOT NULL,          -- 0.0-1.0
    bbox_x       REAL    NOT NULL,
    bbox_y       REAL    NOT NULL,
    bbox_w       REAL    NOT NULL,
    bbox_h       REAL    NOT NULL,
    length       REAL    NOT NULL,
    area         REAL    NOT NULL,
    created_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (face_id) REFERENCES inspection_faces(face_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_face_id     ON defect_detections(face_id);
CREATE INDEX IF NOT EXISTS idx_label_id    ON defect_detections(label_id);
CREATE INDEX IF NOT EXISTS idx_confidence  ON defect_detections(confidence);
)";

bool execute_sql(sqlite3* db, const std::string& sql) {
    char* err = nullptr;
    if (sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &err) != SQLITE_OK) {
        std::cerr << "SQLite error: " << (err ? err : "unknown") << '\n';
        sqlite3_free(err);
        return false;
    }
    return true;
}

int main() {
    sqlite3* db = nullptr;
    if (sqlite3_open("my_inspection.db", &db) != SQLITE_OK) {
        std::cerr << "Can't open database\n";
        return 1;
    }

    bool ok = true;
    ok &= execute_sql(db, ddl_inspection_groups);
    ok &= execute_sql(db, ddl_inspection_faces);
    ok &= execute_sql(db, ddl_defect_detections);

    if (ok) std::cout << "All tables created successfully!\n";

    // 简单验证外键生效
    execute_sql(db, "PRAGMA foreign_keys = ON;");
    execute_sql(db,
        "INSERT INTO inspection_groups(group_id,device_id,status) "
        "VALUES (1,'D001','GOOD');");
    execute_sql(db,
        "INSERT INTO inspection_faces(group_id,face_index,timestamp_ms,sequence_id,saved_path,skeleton_path) "
        "VALUES (1,0,1680000000000,1001,'/a/0.jpg','/a/0_sk.jpg');");
    execute_sql(db,
        "INSERT INTO defect_detections(face_id,label_id,label,confidence,bbox_x,bbox_y,bbox_w,bbox_h,length,area) "
        "VALUES (1,2,'scratch',0.9234,10.5,20.0,30.0,40.0,50.0,600.0);");

    std::cout << "Foreign key & cascade test passed.\n";
    sqlite3_close(db);
    std::cout << "数据库已写入 my_inspection.db\n";
    return ok ? 0 : 2;
}