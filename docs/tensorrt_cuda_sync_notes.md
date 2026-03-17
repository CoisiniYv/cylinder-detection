# TensorRT and OpenCV CUDA Sync Analysis

This note summarizes the current TensorRT inference issues and OpenCV CUDA memory/stream synchronization risks observed in the codebase.

## TensorRT Inference Issues
1. Stream mismatch between memcpy and execution
   - `EngineTRT::infer()` copies inputs with `cudaMemcpyAsync(..., mCudaStream)` but calls `mContext->executeV2(...)` which runs on the default stream.
   - Risk: inference reads incomplete inputs or races with async copies.
   - File: `Core/detect/engineTRT.cpp`
   Code:
   ```cpp
   bool EngineTRT::infer()
   {
       copyInputToDeviceAsync(mCudaStream);
       bool status = mContext->executeV2(mGpuBuffers.data());
       if (!status) { return false; }
       copyOutputToHostAsync(mCudaStream);
       return true;
   }
   ```

2. Output copy without explicit sync
   - `copyOutputToHostAsync(mCudaStream)` is followed by no `cudaStreamSynchronize`.
   - Risk: host reads partial outputs.
   - File: `Core/detect/engineTRT.cpp`
   Code:
   ```cpp
   void EngineTRT::copyOutputToHostAsync(const cudaStream_t& stream)
   {
       memcpyBuffers(false, true, true, stream);
   }
   ```

3. Dynamic input buffer leak
   - `EngineTRT::setInput(float* ...)` allocates new GPU buffers for dynamic inputs but never frees prior allocations.
   - Risk: GPU memory leaks on repeated calls.
   - File: `Core/detect/engineTRT.cpp`
   Code:
   ```cpp
   delete[] mCpuBuffers[1];
   delete[] mCpuBuffers[2];
   mCpuBuffers[1] = new float[numPoints * 2];
   mCpuBuffers[2] = new float[numPoints];

   cudaMalloc(&mGpuBuffers[1], sizeof(float) * numPoints * 2);
   cudaMalloc(&mGpuBuffers[2], sizeof(float) * numPoints);
   ```

4. Dynamic shape setup on different stream than execution
   - `setOptimizationProfileAsync` uses `mCudaStream` but `executeV2` does not.
   - Risk: shape and data readiness not synchronized.
   - File: `Core/detect/engineTRT.cpp`
   Code:
   ```cpp
   mContext->setOptimizationProfileAsync(0, mCudaStream);
   const char* coordsTensorName = mEngine->getIOTensorName(1);
   const char* labelsTensorName = mEngine->getIOTensorName(2);
   mContext->setInputShape(coordsTensorName, nvinfer1::Dims3{ 1, numPoints, 2 });
   mContext->setInputShape(labelsTensorName, nvinfer1::Dims2{ 1, numPoints });
   ```

## OpenCV CUDA Memory/Stream Sync Risks
1. Upload on custom stream, compute on default stream
   - `SingleDetectThread` uses `d_input.upload(src_cpu, stream)` then calls `cropImage(...)` without passing stream.
   - Risk: default stream may read incomplete input.
   - Files: `Core/single_detect_thread.cpp`, `Core/detect/crop_image.cpp`
   Code:
   ```cpp
   cv::cuda::GpuMat d_input;
   d_input.upload(src_cpu, stream);

   cv::cuda::GpuMat d_cropped = cropImage(
       d_input,
       false,
       0, 0, 0, 0,
       false,
       0, 0, 0, 0, 0,
       "fft_image", "",
       p.enable_fourier_transform,
       p.filter_width,
       p.attenuation_factor,
       p.target_angle,
       p.angle_tolerance,
       p.enable_denoising,
       p.denoise_h,
       p.denoise_hColor,
       p.denoise_search_window,
       p.denoise_template_window
   );
   ```

2. StripeRemoval downloads without stream
   - `detect_strongest_direction` calls `download(...)` without stream.
   - Risk: download races with prior GPU ops in a different stream.
   - File: `Core/detect/StripeRemoval.cpp`
   Code:
   ```cpp
   cv::Mat magnitude_spectrum;
   d_magnitude_spectrum.download(magnitude_spectrum);
   ```

3. Halcon and SAM visualization downloads on default stream
   - `gpuMatToHalconHObject` and `SamSegmenter::visualize` call `gpuMat.download(...)` without stream.
   - Risk: downloads can occur before upstream GPU ops finish.
   - Files: `Core/detect/HalconProcessor.cpp`, `Core/detect/sam.cpp`
   Code:
   ```cpp
   // HalconProcessor::gpuMatToHalconHObject
   cv::Mat mat;
   gpuMat.download(mat);
   ```
   ```cpp
   // SamSegmenter::visualize
   cv::Mat image;
   if (!gpuImage.empty()) {
       gpuImage.download(image);
   }
   ```

4. Slice extraction downloads may be async
   - `extractSlicesGPU` does `sliceGPU.download(slice, stream)` and immediately returns for inference.
   - Risk: host may consume slices before download completes.
   - Files: `Core/detect/Slice.cpp`, `Core/detect_thread.cpp`
   Code:
   ```cpp
   cv::Mat slice;
   sliceGPU.download(slice, stream);
   return SliceInfo(slice, cv::Point(startX, startY));
   ```

5. Waiting on the wrong stream
   - `detect_thread.cpp` calls `stream.waitForCompletion()` but some operations (notably default-stream downloads in Halcon/SAM/StripeRemoval) are not covered by that wait.
   - Risk: cross-stream races remain even after waiting on the custom stream.
   - File: `Core/detect_thread.cpp`
   Code:
   ```cpp
   d_cropped.download(vis_cpu, stream);
   stream.waitForCompletion();
   ```

## Directional Fix Suggestions (Not Implemented)
1. Use a single `cv::cuda::Stream` per thread for the full pipeline.
2. Propagate stream into `StripeRemoval`, `HalconProcessor`, `SamSegmenter`, `ImageSlicer` and download only after `stream.waitForCompletion()`.
3. Use TensorRT `enqueueV2`/`enqueueV3` with `mCudaStream`, or synchronize `mCudaStream` before/after `executeV2`.
4. Add explicit `cudaStreamSynchronize` after async output copies.
5. Reuse or free dynamic input GPU buffers in `setInput(float* ...)`.
