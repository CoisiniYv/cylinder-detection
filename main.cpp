#include <iostream>
#include <string>
#include "Core/Server.hpp"
#include "Core/Config.hpp"

using namespace XL;

int main(int argc, char** argv) {
    try {
        const char* cfg_file = nullptr;

        // 命令行参数：-h 打印帮助；-f 指定配置文件路径（默认使用当前目录下的 config.json）
        for (int i = 1; i < argc; i += 2) {
            if (argv[i][0] != '-') {
                printf("参数错误:%s\n", argv[i]);
                return -1;
            }
            switch (argv[i][1]) {
            case 'h': {
                printf("-h 打印参数帮助并退出\n");
                printf("-f 配置文件    如：-f config.json \n");
                return 0;
            }
            case 'f': {
                if (i + 1 < argc) cfg_file = argv[i + 1];
                break;
            }
            default: {
                printf("参数错误:%s\n", argv[i]);
                return -1;
            }
            }
        }

        if (cfg_file == nullptr) {
            cfg_file = "config.json"; // 默认使用当前目录配置
        }

        Config config(cfg_file);
        if (!config.mState) {
            printf("读取配置失败: %s\n", cfg_file);
            return -1;
        }
        config.show();

        // 启动 HTTP Server（阻塞运行，接受 /api/control/add 与 /api/control/cancel 指令）
        Server server;
        server.start(&config);
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
