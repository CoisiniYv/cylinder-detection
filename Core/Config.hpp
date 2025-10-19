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

		std::string ip{};//主机IP地址 10.37.57.112

		int analyzerPort;// 分析服务端口 

		std::string outputdir{};

		std::string modelDir{};


	};
}
