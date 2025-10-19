/*
 同步拍摄示例：演示在指定拍摄次数下，如何进行同步拍摄和图像数据保存
 */
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "MPHdc_API.h"
cv::Mat mergeRGB(unsigned char* ch0,
    unsigned char* ch1,
    unsigned char* ch2,
    int h, int w)
{
    cv::Mat rgb(h, w, CV_8UC3);
    for (int y = 0; y < h; ++y) {
        cv::Vec3b* row = rgb.ptr<cv::Vec3b>(y);
        for (int x = 0; x < w; ++x) {
            row[x][0] = ch0[y * w + x];   // B
            row[x][1] = ch1[y * w + x];   // G
            row[x][2] = ch2[y * w + x];   // R
        }
    }
    return rgb;
}
int main() {
    setlocale(LC_ALL, "");

    mphdc::IMPHdc* Device;
    Device = mphdc::MPHdc_Factory::GetInstance(mphdc::LogMediaType::CallBack);

    // 连接设备前先更新信息
    Device->UpdateDeviceList();
    int TotalDeviceCnt = Device->GetDeviceCount();

    // 只打开第一个设备
    bool rtv = Device->Open(Device->GetDeviceInfo(0));
    if (!rtv) {
        std::cout << std::endl << "Open device failed." << std::endl;
        mphdc::MPHdc_Factory::DestructInstance(Device);
        system("pause");
        return 0;
    }

    std::cout << std::endl << "Open device successfully." << std::endl;

    // 输出当前模式
    mphdc::BasicSettingsStructType BasicSettings;
    Device->GetBasicSettings(&BasicSettings);
    std::cout << "当前相机工作模式：" << mphdc::EnumUtils::GetEnumName(BasicSettings.WorkingMode.Mode) << std::endl;

    // 关闭hold状态
    BasicSettings.HoldState = 0;
    // 打开软触发
    BasicSettings.TriggerSource = mphdc::TriggerSourceType::SoftTriggerOnly;
    // 保存设置
    Device->SetBasicSettings(BasicSettings);
    // 受限于相机机制，对于写入设置这个过程并没有一个状态来表示，导致不知道设置什么时候写完
    // 进而导致当设置没完全写入时，开始拍摄就会出现问题
    // 所以有必要的话可以加一个Sleep
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    mphdc::DataFormatType format;
    mphdc::DataFrameUndefinedStruct data;
    // 同步拍摄

    for (int i = 0; i < 1; i++) {
        if (!Device->Snap(true, &format, &data, 10000)) {
            std::cout << "拍摄失败" << std::endl;
            continue;
        }
        std::cout << "获取图像格式" << mphdc::EnumUtils::GetEnumName(data.FrameInfo.DataInfo.DataFormat) << " 数量"
            << (uint32_t)data.FrameInfo.DataInfo.DataNumber << std::endl;

        // 强制转换为DataFrame2DStruct来使用更多方法
        mphdc::DataFrame2DStruct* data2D = reinterpret_cast<mphdc::DataFrame2DStruct*>(&data);

        const mphdc::ChannelContentSequenceStructType& channelContentSequence
            = data2D->FrameInfo.DataInfo.ChannelContentSequence;
        std::array<unsigned char*, 6> chPtr{};  // 保存 6 个通道的指针
        // 遍历原始单通道
        int height = data2D->Height();   // 图像高
        int width = data2D->Width();
        for (int i = 0; i < data2D->Channel(); ++i) {

            unsigned char* ptr = nullptr;
            int size = data2D->GetRawChannelData(i, &ptr);
            if (ptr && size == height * width) {   // 简单校验
                chPtr[i] = ptr;
                std::cout << "通道 " << i << " 已就绪\n";
            }
        }

        // 拼第一张 RGB（0,1,2）
        if (chPtr[2] && chPtr[1] && chPtr[0]) {
            cv::Mat rgb1 = mergeRGB(chPtr[2], chPtr[1], chPtr[0], height, width);
            cv::imwrite("rgb_012.png", rgb1);
            std::cout << "已写入 rgb_012.png\n";
        }

        // 拼第二张 RGB（3,4,5）
        if (chPtr[5] && chPtr[4] && chPtr[3]) {
            cv::Mat rgb2 = mergeRGB(chPtr[5], chPtr[4], chPtr[3], height, width);
            cv::imwrite("rgb_345.png", rgb2);
            std::cout << "已写入 rgb_345.png\n";
        }
    }

    // 最后保存数据，实际上是最后一次拍到的图
    // mphdc::MPHdc_Utils::SaveDataFrame(data, "./test.mphdcdat", 0, nullptr);

    //Utils.SaveDataFrame(data, "./test.mphdcdat", 0, IntPtr.Zero);
    //data.GetGrayBitmap().Save("./testpng.png", System.Drawing.Imaging.ImageFormat.Png);

    // 打开hold状态
    BasicSettings.HoldState = 1;
    // 保存设置
    Device->SetBasicSettings(BasicSettings);

    // 关闭连接
    Device->Close();
    // 释放Device，注意不能直接delete
    mphdc::MPHdc_Factory::DestructInstance(Device);

    // 暂停
    fflush(stdin);
    getchar();

    return 0;
}
