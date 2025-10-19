#include "Config.hpp"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <json/json.h>
#include "Utils/Log.hpp"

namespace XL {
    Config::Config(const char* file) :
        file(file)
    {
        std::ifstream ifs(file, std::ios::binary);
        if (!ifs.is_open()) {
            LOGE("open %s error", file);
            return;
        }
        else {
            Json::CharReaderBuilder builder;
            builder["collectComments"] = true;
            JSONCPP_STRING errs;
            Json::Value root;

            if (parseFromStream(builder, ifs, &root, &errs)) {
                // 兼容两种字段名：优先 host；否则使用 ip
                this->ip = root.isMember("host") ? root["host"].asString() : root.get("ip", "").asString();
                this->analyzerPort = root.get("analyzerPort", 0).asInt();

                this->outputdir = root.get("uploadDir", "").asString();
                this->modelDir = root.get("modelDir", "").asString();
                this->dbPath   = root.get("dbPath", "").asString();

                std::filesystem::path path(outputdir);
                try {
                    if (!std::filesystem::exists(path)) {
                        std::filesystem::create_directory(path);
                    }
                    mState = true;
                }
                catch (std::filesystem::filesystem_error& e) {
                    std::cout << e.what() << std::endl;
                }
            }
            else {
                LOGE("parse %s error", file);
            }
            ifs.close();
        }
    }

    Config::~Config()
    {

    }

    void Config::show() {
        printf("config.file=%s\n", file);
        printf("config.ip=%s\n", ip.c_str());
        printf("config.analyzerPort=%d\n", analyzerPort);
        printf("config.uploadDir=%s\n", outputdir.data());
        printf("config.modelDir=%s\n", modelDir.data());
        printf("config.dbPath=%s\n", dbPath.empty() ? "(empty)" : dbPath.c_str());
    }
}