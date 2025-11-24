#pragma once

#include "macros.h"
#include "NvInferRuntimeCommon.h"

#include <cassert>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>

class LogStreamConsumerBuffer : public std::stringbuf
{
public:
	LogStreamConsumerBuffer(std::ostream& stream, const std::string& prefix, bool shouldLog)
		: mOutput(stream)
		, mPrefix(prefix)
		, mShouldLog(shouldLog)
	{
	}

	LogStreamConsumerBuffer(LogStreamConsumerBuffer&& other)
		: mOutput(other.mOutput)
	{
	}

	~LogStreamConsumerBuffer()
	{
		// std::streambuf::pbase() 返回输出序列缓冲部分起始位置的指针
		// std::streambuf::pptr() 返回输出序列当前位置的指针
		// 如果起始位置指针不等于当前位置指针，调用 putOutput() 将输出记录到流中
		if (pbase() != pptr())
		{
			putOutput();
		}
	}

	// 同步流缓冲区，成功时返回0
	// 同步流缓冲区包括将缓冲区内容插入到流中、重置缓冲区并刷新流
	virtual int sync()
	{
		putOutput();
		return 0;
	}

	void putOutput()
	{
		if (mShouldLog)
		{
			// 添加时间戳前缀
			std::time_t timestamp = std::time(nullptr);
			std::tm tm_local{};
#if defined(_MSC_VER)                  // Visual Studio
			localtime_s(&tm_local, &timestamp);
#else                                  // gcc/clang
			localtime_r(&timestamp, &tm_local);
#endif
			std::cout << '['
				<< std::setw(2) << std::setfill('0') << tm_local.tm_mon + 1 << '/'
				<< std::setw(2) << std::setfill('0') << tm_local.tm_mday << '/'
				<< std::setw(4) << tm_local.tm_year + 1900 << '-'
				<< std::setw(2) << std::setfill('0') << tm_local.tm_hour << ':'
				<< std::setw(2) << std::setfill('0') << tm_local.tm_min << ':'
				<< std::setw(2) << std::setfill('0') << tm_local.tm_sec << "] ";

			// std::stringbuf::str() 获取缓冲区的字符串内容
			// 将缓冲区内容（前面加上适当的前缀）插入到流中
			mOutput << mPrefix << str();
			// 将缓冲区设置为空
			str("");
			// 刷新流
			mOutput.flush();
		}
	}

	void setShouldLog(bool shouldLog)
	{
		mShouldLog = shouldLog;
	}

private:
	std::ostream& mOutput;
	std::string mPrefix;
	bool mShouldLog;
};

//!
//! \class LogStreamConsumerBase
//! \brief 用于在 LogStreamConsumer 中的 std::ostream 之前初始化 LogStreamConsumerBuffer 的便利对象
//!
class LogStreamConsumerBase
{
public:
	LogStreamConsumerBase(std::ostream& stream, const std::string& prefix, bool shouldLog)
		: mBuffer(stream, prefix, shouldLog)
	{
	}

protected:
	LogStreamConsumerBuffer mBuffer;
};

//!
//! \class LogStreamConsumer
//! \brief 用于在使用流语法记录消息时提供便利的对象。
//!  基类的顺序是 LogStreamConsumerBase，然后是 std::ostream。
//!  这是因为 LogStreamConsumerBase 类用于初始化 LogStreamConsumer 中的 LogStreamConsumerBuffer 成员字段，
//!  然后将缓冲区的地址传递给 std::ostream。
//!  这是为了避免将未初始化的缓冲区地址传递给 std::ostream。
//!  请不要更改父类的顺序。
//!
class LogStreamConsumer : protected LogStreamConsumerBase, public std::ostream
{
public:
	//! \brief 创建一个 LogStreamConsumer，用于记录具有指定级别的消息。
	//!  可报告级别决定消息是否足够严重以至于需要被记录。
	LogStreamConsumer(nvinfer1::ILogger::Severity reportableSeverity, nvinfer1::ILogger::Severity severity)
		: LogStreamConsumerBase(severityOstream(severity), severityPrefix(severity), severity <= reportableSeverity)
		, std::ostream(&mBuffer) // 将流缓冲区与流链接
		, mShouldLog(severity <= reportableSeverity)
		, mSeverity(severity)
	{
	}

	LogStreamConsumer(LogStreamConsumer&& other)
		: LogStreamConsumerBase(severityOstream(other.mSeverity), severityPrefix(other.mSeverity), other.mShouldLog)
		, std::ostream(&mBuffer) // 将流缓冲区与流链接
		, mShouldLog(other.mShouldLog)
		, mSeverity(other.mSeverity)
	{
	}

	void setReportableSeverity(nvinfer1::ILogger::Severity reportableSeverity)
	{
		mShouldLog = mSeverity <= reportableSeverity;
		mBuffer.setShouldLog(mShouldLog);
	}

private:
	static std::ostream& severityOstream(nvinfer1::ILogger::Severity severity)
	{
		return severity >= nvinfer1::ILogger::Severity::kINFO ? std::cout : std::cerr;
	}

	static std::string severityPrefix(nvinfer1::ILogger::Severity severity)
	{
		switch (severity)
		{
		case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "[F] ";
		case nvinfer1::ILogger::Severity::kERROR: return "[E] ";
		case nvinfer1::ILogger::Severity::kWARNING: return "[W] ";
		case nvinfer1::ILogger::Severity::kINFO: return "[I] ";
		case nvinfer1::ILogger::Severity::kVERBOSE: return "[V] ";
		default: assert(0); return "";
		}
	}

	bool mShouldLog;
	nvinfer1::ILogger::Severity mSeverity;
};

//! \class Logger
//!
//! \brief 管理 TensorRT 工具和示例日志记录的类
//!
//! \details 该类为 TensorRT 工具和示例提供了一个统一的接口，用于将信息记录到控制台，
//! 并支持记录两种类型的消息：
//!
//! - 带有相关严重级别（信息、警告、错误或内部错误/致命）的调试消息
//! - 测试通过/失败消息
//!
//! 让所有示例使用此类进行记录，而不是直接输出到 stdout/stderr 的好处在于，
//! 控制示例输出详细程度和格式的逻辑集中在一个位置。
//!
//! 将来，可以扩展此类以支持将测试结果转储到某种标准格式的文件中（例如 JUnit XML），
//! 并提供额外的元数据（例如测试运行的持续时间）。
//!
//! TODO：为了与现有示例向后兼容，此类直接继承自 nvinfer1::ILogger
//! 接口，这存在一个问题，因为来自 TensorRT 库的消息和来自示例的消息之间没有清晰的分离。
//!
//! 将来（一旦所有示例都更新为使用 Logger::getTRTLogger() 来访问 ILogger）我们可以重构
//! 该类以消除继承，而是将 nvinfer1::ILogger 实现作为 Logger 对象的成员。

class Logger : public nvinfer1::ILogger
{
public:
	Logger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kWARNING)
		: mReportableSeverity(severity)
	{
	}

	//!
	//! \enum TestResult
	//! \brief 表示给定测试的状态
	//!
	enum class TestResult
	{
		kRUNNING, //!< 测试正在运行
		kPASSED,  //!< 测试通过
		kFAILED,  //!< 测试失败
		kWAIVED   //!< 测试被豁免
	};

	//!
	//! \brief 用于检索与此 Logger 关联的 nvinfer::ILogger 的向前兼容方法
	//! \return 与此 Logger 关联的 nvinfer1::ILogger
	//!
	//! TODO 一旦所有示例都更新为使用此方法向 TensorRT 注册记录器，
	//! 我们可以消除 Logger 从 ILogger 的继承
	//!
	nvinfer1::ILogger& getTRTLogger()
	{
		return *this;
	}

	//!
	//! \brief nvinfer1::ILogger::log() 虚方法的实现
	//!
	//! 注意示例不应直接调用此函数；一旦我们消除从 nvinfer1::ILogger 的继承，它最终将会消失
	//!
	void log(nvinfer1::ILogger::Severity severity, const char* msg) TRT_NOEXCEPT override
	{
		LogStreamConsumer(mReportableSeverity, severity) << "[TRT] " << std::string(msg) << std::endl;
	}

	//!
	//! \brief 用于控制日志记录输出详细程度的方法
	//!
	//! \param severity 记录器只会发出严重级别等于或高于此级别的消息。
	//!
	void setReportableSeverity(nvinfer1::ILogger::Severity severity)
	{
		mReportableSeverity = severity;
	}

	//!
	//! \brief 用于保存特定测试日志记录信息的不透明句柄
	//!
	//! 此对象是 Logger 用于打印测试结果信息的不透明句柄。
	//! 示例必须调用 Logger::defineTest() 才能获得一个可用于
	//! Logger::reportTest{Start,End}() 的 TestAtom。
	//!
	class TestAtom
	{
	public:
		TestAtom(TestAtom&&) = default;

	private:
		friend class Logger;

		TestAtom(bool started, const std::string& name, const std::string& cmdline)
			: mStarted(started)
			, mName(name)
			, mCmdline(cmdline)
		{
		}

		bool mStarted;
		std::string mName;
		std::string mCmdline;
	};

	//!
	//! \brief 为日志记录定义一个测试
	//!
	//! \param[in] name 测试的名称。这应该是一个以 "TensorRT" 开头并包含点分隔字符串的字符串，
	//!                  这些字符串包含字符 [A-Za-z0-9_]。
	//!                  例如："TensorRT.sample_googlenet"
	//! \param[in] cmdline 用于重现测试的命令行
	//
	//! \return 一个可用于 Logger::reportTest{Start,End}() 的 TestAtom。
	//!
	static TestAtom defineTest(const std::string& name, const std::string& cmdline)
	{
		return TestAtom(false, name, cmdline);
	}

	//!
	//! \brief defineTest() 的一个便利重载版本，接受命令行参数数组作为输入
	//!
	//! \param[in] name 测试的名称
	//! \param[in] argc 命令行参数的数量
	//! \param[in] argv 命令行参数数组（以 C 字符串形式给出）
	//!
	//! \return 一个可用于 Logger::reportTest{Start,End}() 的 TestAtom。
	static TestAtom defineTest(const std::string& name, int argc, char const* const* argv)
	{
		auto cmdline = genCmdlineString(argc, argv);
		return defineTest(name, cmdline);
	}

	//!
	//! \brief 报告测试已开始。
	//!
	//! \pre 尚未对给定的 testAtom 调用 reportTestStart()
	//!
	//! \param[in] testAtom 已开始测试的句柄
	//!
	static void reportTestStart(TestAtom& testAtom)
	{
		reportTestResult(testAtom, TestResult::kRUNNING);
		assert(!testAtom.mStarted);
		testAtom.mStarted = true;
	}

	//!
	//! \brief 报告测试已结束。
	//!
	//! \pre 已对给定的 testAtom 调用 reportTestStart()
	//!
	//! \param[in] testAtom 已结束测试的句柄
	//! \param[in] result 测试的结果。应为 TestResult::kPASSED、
	//!                   TestResult::kFAILED、TestResult::kWAIVED 之一
	//!
	static void reportTestEnd(const TestAtom& testAtom, TestResult result)
	{
		assert(result != TestResult::kRUNNING);
		assert(testAtom.mStarted);
		reportTestResult(testAtom, result);
	}

	static int reportPass(const TestAtom& testAtom)
	{
		reportTestEnd(testAtom, TestResult::kPASSED);
		return EXIT_SUCCESS;
	}

	static int reportFail(const TestAtom& testAtom)
	{
		reportTestEnd(testAtom, TestResult::kFAILED);
		return EXIT_FAILURE;
	}

	static int reportWaive(const TestAtom& testAtom)
	{
		reportTestEnd(testAtom, TestResult::kWAIVED);
		return EXIT_SUCCESS;
	}

	static int reportTest(const TestAtom& testAtom, bool pass)
	{
		return pass ? reportPass(testAtom) : reportFail(testAtom);
	}

	nvinfer1::ILogger::Severity getReportableSeverity() const
	{
		return mReportableSeverity;
	}

private:
	//!
	//! \brief 返回一个适当的字符串，用于为具有给定严重级别的日志消息添加前缀
	//!
	static const char* severityPrefix(nvinfer1::ILogger::Severity severity)
	{
		switch (severity)
		{
		case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "[F] ";
		case nvinfer1::ILogger::Severity::kERROR: return "[E] ";
		case nvinfer1::ILogger::Severity::kWARNING: return "[W] ";
		case nvinfer1::ILogger::Severity::kINFO: return "[I] ";
		case nvinfer1::ILogger::Severity::kVERBOSE: return "[V] ";
		default: assert(0); return "";
		}
	}

	//!
	//! \brief 返回一个适当的字符串，用于为具有给定结果的测试结果消息添加前缀
	//!
	static const char* testResultString(TestResult result)
	{
		switch (result)
		{
		case TestResult::kRUNNING: return "RUNNING";
		case TestResult::kPASSED: return "PASSED";
		case TestResult::kFAILED: return "FAILED";
		case TestResult::kWAIVED: return "WAIVED";
		default: assert(0); return "";
		}
	}

	//!
	//! \brief 返回与给定严重级别一起使用的适当输出流（cout 或 cerr）
	//!
	static std::ostream& severityOstream(nvinfer1::ILogger::Severity severity)
	{
		return severity >= nvinfer1::ILogger::Severity::kINFO ? std::cout : std::cerr;
	}

	//!
	//! \brief 实现记录测试结果的方法
	//!
	static void reportTestResult(const TestAtom& testAtom, TestResult result)
	{
		severityOstream(nvinfer1::ILogger::Severity::kINFO) << "&&&& " << testResultString(result) << " " << testAtom.mName << " # "
			<< testAtom.mCmdline << std::endl;
	}

	//!
	//! \brief 从给定的 (argc, argv) 值生成命令行字符串
	//!
	static std::string genCmdlineString(int argc, char const* const* argv)
	{
		std::stringstream ss;
		for (int i = 0; i < argc; i++)
		{
			if (i > 0)
				ss << " ";
			ss << argv[i];
		}
		return ss.str();
	}

	nvinfer1::ILogger::Severity mReportableSeverity;
};

namespace
{

	//!
	//! \brief 生成一个 LogStreamConsumer 对象，可用于记录严重级别为 kVERBOSE 的消息
	//!
	//! 用法示例：
	//!
	//!     LOG_VERBOSE(logger) << "hello world" << std::endl;
	//!
	inline LogStreamConsumer LOG_VERBOSE(const Logger& logger)
	{
		return LogStreamConsumer(logger.getReportableSeverity(), nvinfer1::ILogger::Severity::kVERBOSE);
	}

	//!
	//! \brief 生成一个 LogStreamConsumer 对象，可用于记录严重级别为 kINFO 的消息
	//!
	//! 用法示例：
	//!
	//!     LOG_INFO(logger) << "hello world" << std::endl;
	//!
	inline LogStreamConsumer LOG_INFO(const Logger& logger)
	{
		return LogStreamConsumer(logger.getReportableSeverity(), nvinfer1::ILogger::Severity::kINFO);
	}

	//!
	//! \brief 生成一个 LogStreamConsumer 对象，可用于记录严重级别为 kWARNING 的消息
	//!
	//! 用法示例：
	//!
	//!     LOG_WARN(logger) << "hello world" << std::endl;
	//!
	inline LogStreamConsumer LOG_WARN(const Logger& logger)
	{
		return LogStreamConsumer(logger.getReportableSeverity(), nvinfer1::ILogger::Severity::kWARNING);
	}

	//!
	//! \brief 生成一个 LogStreamConsumer 对象，可用于记录严重级别为 kERROR 的消息
	//!
	//! 用法示例：
	//!
	//!     LOG_ERROR(logger) << "hello world" << std::endl;
	//!
	inline LogStreamConsumer LOG_ERROR(const Logger& logger)
	{
		return LogStreamConsumer(logger.getReportableSeverity(), nvinfer1::ILogger::Severity::kERROR);
	}

	//!
	//! \brief 生成一个 LogStreamConsumer 对象，可用于记录严重级别为 kINTERNAL_ERROR
	//!        （"致命"严重级别）的消息
	//!
	//! 用法示例：
	//!
	//!     LOG_FATAL(logger) << "hello world" << std::endl;
	//!
	inline LogStreamConsumer LOG_FATAL(const Logger& logger)
	{
		return LogStreamConsumer(logger.getReportableSeverity(), nvinfer1::ILogger::Severity::kINTERNAL_ERROR);
	}

} // 匿名命名空间