#pragma once

#include "detect/HalconProcessor.h"
#include "detect/Slice.h"
#include "detect/crop_image.h"
#include "detect/sam.h"
#include "detect/trtyolo_slice.hpp"
#include "detect_params.hpp"

#include <atomic>
#include <memory>
#include <thread>

namespace XL {

// One GPU worker in the multi-face pipeline.
//
// Responsibilities: own thread-local model instances, consume ImageFrame from
// QueueManager, execute the detection pipeline and publish SingleImageResult.
class DetectThread {
public:
    DetectThread(int thread_index, const DetectParams& params);
    ~DetectThread();

    DetectThread(const DetectThread&) = delete;
    DetectThread& operator=(const DetectThread&) = delete;

    void start();
    void stop();
    void join();
    bool isRunning() const noexcept { return mRunning.load(std::memory_order_relaxed); }
    int index() const noexcept { return mIndex; }

private:
    void run();

private:
    int mIndex = 0;
    std::thread mThread;
    std::atomic<bool> mRunning{false};
    DetectParams mParams;

    std::unique_ptr<trtyolo::SliceDetector> mSliceDetector;
    HalconProcessor mHalcon;
    bool mModelReady = false;

    std::unique_ptr<SamSegmenter> mSam;
    bool mSamReady = false;
    bool mPreferPsImage = true;
};

} // namespace XL
