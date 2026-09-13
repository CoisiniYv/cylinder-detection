#pragma once

#include <string>

namespace XL {

class Config {
public:
    explicit Config(const char* file);
    ~Config() = default;

    void show() const;

public:
    bool mState = false;
    std::string sourceFile;

    std::string ip{"127.0.0.1"};
    int analyzerPort = 9003;

    std::string outputdir{"output"};
    std::string modelDir{"models"};
    std::string dbPath{"my_inspection.db"};

    std::string slidePort{};
    int slideAxisId = 0;
    int slideTimeoutMs = 20000;

    int gpuDevice = 0;
    int detectThreads = 4;
    int groupTimeoutMs = 55000;
};

} // namespace XL
