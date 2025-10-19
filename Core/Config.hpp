#pragma once
#include <string>
#include <vector>

namespace XL{
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


	};
}
