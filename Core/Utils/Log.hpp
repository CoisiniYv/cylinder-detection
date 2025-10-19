#pragma once
#include <time.h>
#include <string>
namespace XL{
#pragma warning( disable : 4996 )

    static std::string XLlogTime() {
        const char* time_fmt = "%Y-%m-%d %H:%M:%S";
        time_t t = time(nullptr);
        char time_str[64];
        strftime(time_str, sizeof(time_str), time_fmt, localtime(&t));

        return time_str;
    }




#define LOGI(format, ...)  fprintf(stderr,"[INFO]%s [%s:%d] " format "\n", XL::XLlogTime().data(),__func__,__LINE__,##__VA_ARGS__)
#define LOGE(format, ...)  fprintf(stderr,"[ERROR]%s [%s:%d] " format "\n",XL::XLlogTime().data(),__func__,__LINE__,##__VA_ARGS__)
}