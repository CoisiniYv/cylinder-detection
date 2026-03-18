#pragma once

#include <string>
#include <vector>

namespace XL {
	class Config
	{
	public:
		Config(const char* file);
		~Config();
	public:

		bool mState = false;
		void show();
	public:
		const char* file = NULL;

		std::string ip{};//IP地址10.37.57.112

		int analyzerPort;//服务器端口 

		std::string outputdir{};

		std::string modelDir{};

		std::string dbPath{}; // 数据库文件路径（例如 E:\ydk\resource\my_inspection.db）

		// 滑台串口配置（UART1）
		std::string slidePort{}; // 例如 "COM11" 或 "\\\\.\\COM11"
		int slideAxisId = 0;     // 轴ID（推荐 0=X,1=Y,2=Z）
		int slideTimeoutMs = 20000; // 等待下位机回报超时

	};
}
