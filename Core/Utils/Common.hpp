#pragma once
#ifndef XL_COMMON_H
#define XL_COMMON_H
#include <string>
#include <vector>
#include <chrono>
#include <ctime>
#include <string>

namespace XL
{
    static int64_t getCurTime()// 获取当前系统启动以来的毫秒数
    {
#ifdef _WIN32               // 肯定存在
#  include <chrono>
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
#else
#  include <time.h>
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return int64_t(ts.tv_sec) * 1000 + ts.tv_nsec / 1000000;
#endif

    }
    static int64_t getCurTimestamp()// 获取毫秒级时间戳（13位）
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).
            count();

    }
    static std::string getCurFormatTimeStr(const char* format = "%Y-%m-%d %H:%M:%S")
    {
        std::string ret(64, '\0');

        std::time_t t = std::time(nullptr);
        std::tm tmb{};

        #if defined(_WIN32)
                localtime_s(&tmb, &t);                 // MSVC 安全版
        #else
                localtime_r(&t, &tmb);                 // POSIX 线程安全版
        #endif

                std::size_t len = std::strftime(&ret[0], ret.size(), format, &tmb);
                ret.resize(len);                       // 去掉多余 \0
                return ret;
    }
    static std::vector<std::string> split(const std::string& str, const std::string& sep) {

        std::vector<std::string> arr;
        int sepSize = sep.size();

        int lastPosition = 0, index = -1;
        while (-1 != (index = str.find(sep, lastPosition)))
        {
            arr.push_back(str.substr(lastPosition, index - lastPosition));
            lastPosition = index + sepSize;
        }
        std::string lastStr = str.substr(lastPosition);//截取最后一个分隔符后的内容

        if (!lastStr.empty()) {
            arr.push_back(lastStr);//如果最后一个分隔符后还有内容就入队
        }

        return arr;
    }

    static bool removeFile(const std::string& filename) {

        if (remove(filename.data()) == 0) {
            return true;
        }
        else {
            return false;
        }
    }

    static int getRandomInt() {
        std::string numStr;
        numStr.append(std::to_string(std::rand() % 9 + 1));
        numStr.append(std::to_string(std::rand() % 10));
        numStr.append(std::to_string(std::rand() % 10));
        numStr.append(std::to_string(std::rand() % 10));
        numStr.append(std::to_string(std::rand() % 10));
        numStr.append(std::to_string(std::rand() % 10));
        numStr.append(std::to_string(std::rand() % 10));
        numStr.append(std::to_string(std::rand() % 10));
        int num = stoi(numStr);

        return num;
    }


};
#endif 