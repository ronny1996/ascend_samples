#pragma once
#include "npu_internal_helper.h"

struct NpuDeviceHelper {
  static void ExceptionCallback(aclrtExceptionInfo* exceptionInfo) {
    std::cerr << "Enter NPU ExceptionCallback\n";
    auto deviceId = aclrtGetDeviceIdFromExceptionInfo(exceptionInfo);
    auto streamId = aclrtGetStreamIdFromExceptionInfo(exceptionInfo);
    auto taskId = aclrtGetTaskIdFromExceptionInfo(exceptionInfo);

    char opName[256];
    aclTensorDesc* inputDesc = nullptr;
    aclTensorDesc* outputDesc = nullptr;
    size_t inputCnt = 0;
    size_t outputCnt = 0;
    aclmdlCreateAndGetOpDesc(deviceId, streamId, taskId, opName, 256,
                             &inputDesc, &inputCnt, &outputDesc, &outputCnt);
    for (size_t i = 0; i < inputCnt; ++i) {
      const aclTensorDesc* desc = aclGetTensorDescByIndex(inputDesc, i);
      aclGetTensorDescAddress(desc);
      aclGetTensorDescFormat(desc);
    }
    for (size_t i = 0; i < outputCnt; ++i) {
      const aclTensorDesc* desc = aclGetTensorDescByIndex(outputDesc, i);
      aclGetTensorDescAddress(desc);
      aclGetTensorDescFormat(desc);
    }
    aclDestroyTensorDesc(inputDesc);
    aclDestroyTensorDesc(outputDesc);
    std::cerr << "Exit NPU ExceptionCallback\n";
  }

  static void GetRecentErrorMessage() {
    const char* aclRecentErrMsg = nullptr;
    aclRecentErrMsg = aclGetRecentErrMsg();
    if (aclRecentErrMsg != nullptr) {
      printf("%s\n", aclRecentErrMsg);
    } else {
      printf("Failed to get recent error message.\n");
    }
  }

  // Device
  static void InitAllDevices(const std::vector<int32_t>& devices) {
    ACL_CHECK(aclInit(nullptr));
    std::cout << "aclInit()\n";
    for (auto dev_id : devices) {
      assert(dev_id >= 0 && dev_id < NpuDeviceHelper::GetDevicesCount());
      ACL_CHECK(aclrtSetDevice(dev_id));
      std::cout << "aclrtSetDevice(" << dev_id << ")\n";
    }
    ACL_CHECK(
        aclrtSetExceptionInfoCallback(NpuDeviceHelper::ExceptionCallback));
    std::cout << "aclrtSetExceptionInfoCallback("
              << &NpuDeviceHelper::ExceptionCallback << ")\n";
  }

  static void InitAllDevices() {
    ACL_CHECK(aclInit(nullptr));
    std::cout << "aclInit()\n";
    for (auto dev_id = 0; dev_id < NpuDeviceHelper::GetDevicesCount();
         dev_id++) {
      ACL_CHECK(aclrtSetDevice(dev_id));
      std::cout << "aclrtSetDevice(" << dev_id << ")\n";
    }
    ACL_CHECK(
        aclrtSetExceptionInfoCallback(NpuDeviceHelper::ExceptionCallback));
    std::cout << "aclrtSetExceptionInfoCallback("
              << &NpuDeviceHelper::ExceptionCallback << ")\n";
  }

  static void ReleaseAllDevices(const std::vector<int32_t>& devices) {
    for (auto dev_id : devices) {
      assert(dev_id >= 0 && dev_id < NpuDeviceHelper::GetDevicesCount());
      ACL_CHECK(aclrtResetDevice(dev_id));
      std::cout << "aclrtResetDevice(" << dev_id << ")\n";
    }
    ACL_CHECK(aclFinalize());
    std::cout << "aclFinalize()\n";
  }

  static void ReleaseAllDevices() {
    for (auto dev_id = 0; dev_id < NpuDeviceHelper::GetDevicesCount();
         dev_id++) {
      ACL_CHECK(aclrtResetDevice(dev_id));
      std::cout << "aclrtResetDevice(" << dev_id << ")\n";
    }
    ACL_CHECK(aclFinalize());
    std::cout << "aclFinalize()\n";
  }

  static void SetCurrentDevice(int dev_id) {
    if (dev_id > NpuDeviceHelper::GetDevicesCount() || dev_id < 0) {
      std::cerr << "dev_id > NpuDeviceHelper::GetDevicesCount() || dev_id < 0"
                << std::endl;
    } else {
      aclrtSetDevice(dev_id);
      std::cout << "aclrtSetDevice(" << dev_id << ")\n";
    }
  }

  static uint32_t GetDevicesCount() {
    uint32_t count;
    ACL_CHECK(aclrtGetDeviceCount(&count));
    return count;
  }

  static int GetCurrentDeivce() {
    int dev_id;
    ACL_CHECK(aclrtGetDevice(&dev_id));
    return dev_id;
  }

  std::string GetRunMode() {
    aclrtRunMode run_mode;
    ACL_CHECK(aclrtGetRunMode(&run_mode));
    for (const auto& pair : string_run_mode_mapping) {
      if (pair.second == run_mode) {
        return pair.first;
      }
    }
    std::cerr << "[GetRunMode] cant find " << static_cast<int>(run_mode)
              << std::endl;
    exit(0);
  }

  // Context
  aclrtContext CreateContext(int32_t dev_id) {
    aclrtContext context;
    ACL_CHECK(aclrtCreateContext(&context, dev_id));
    return context;
  }

  void DestroyContext(aclrtContext context) {
    ACL_CHECK(aclrtDestroyContext(context));
  }

  aclrtContext GetCurrentContext() {
    aclrtContext context;
    ACL_CHECK(aclrtGetCurrentContext(&context));
    return context;
  }

  void SetCurrentContext(aclrtContext context) {
    ACL_CHECK(aclrtSetCurrentContext(context));
  }

  aclrtStream CreateStream() {
    aclrtStream stream;
    ACL_CHECK(aclrtCreateStream(&stream));
    return stream;
  }

  void DestroyStream(aclrtStream stream) {
    ACL_CHECK(aclrtDestroyStream(stream));
  }

  /**
    aclrtCreateEvent
    aclrtDestroyEvent
    aclrtRecordEvent
    aclrtResetEvent
    aclrtQueryEvent
    aclrtSynchronizeEvent
    aclrtEventElapsedTime
    aclrtStreamWaitEvent
    aclrtSynchronizeDevice
    aclrtSynchronizeStream
    aclrtSubscribeReport
    aclrtLaunchCallback
    aclrtProcessReport
    aclrtUnSubscribeReport
    aclrtSetExceptionInfoCallback
    aclrtGetTaskIdFromExceptionInfo
    aclrtGetStreamIdFromExceptionInfo
    aclrtGetThreadIdFromExceptionInfo
    aclrtGetDeviceIdFromExceptionInfo
  */

  /*
    aclrtMalloc
    aclrtMallocCached
    aclrtMemFlush
    aclrtMemInvalidate
    aclrtFree
    aclrtMallocHost
    aclrtFreeHost
    aclrtMemset
    aclrtMemsetAsync
    aclrtMemcpy
    aclrtMemcpyAsync
    aclrtGetMemInfo
  */
};

struct NpuStream {
  NpuStream() { ACL_CHECK(aclrtCreateStream(&_stream)); }

  NpuStream(aclrtStream stream) : _stream(stream) {}

  ~NpuStream() {
    this->Wait();
    if (_stream) {
      ACL_CHECK(aclrtDestroyStream(_stream));
      _stream = nullptr;
    }
  }

  void Wait() {
    if (_stream) {
      ACL_CHECK(aclrtSynchronizeStream(_stream));
    }
  }

  aclrtStream GetStream() { return _stream; }

 private:
  aclrtStream _stream = nullptr;
};

struct NpuContext {
  NpuContext(int dev_id) { ACL_CHECK(aclrtCreateContext(&_context, dev_id)); }

  NpuContext(aclrtContext context) : _context(context) {}

  ~NpuContext() {
    this->Wait();
    if (_context) {
      ACL_CHECK(aclrtDestroyContext(_context));
      _context = nullptr;
    }
  }

  void Wait() {
    if (_context) {
    }
  }

  aclrtContext GetContext() { return _context; }

  void SetCurrentContext() { ACL_CHECK(aclrtSetCurrentContext(_context)); }

 private:
  aclrtContext _context = nullptr;
};

struct NpuDevice {
  NpuDevice(int dev_id) : _dev_id(dev_id) {}

  int GetId() { return _dev_id; }

  void Wait() {
    aclrtSynchronizeDevice();  // wait for device which according to the current
                               // context
  }

  NpuContext CreateContext() { return NpuContext(_dev_id); }

  NpuStream CreateStream() { return NpuStream(); }

  void SetCurrentDevice() {
    NpuDeviceHelper::SetCurrentDevice(_dev_id);
  }

  // aclrtDeviceCanAccessPeer
  // aclrtDeviceEnablePeerAccess
  // aclrtDeviceDisablePeerAccess
 private:
  int _dev_id = 0;
};

struct NpuGuard {
  NpuGuard(int dev_id) {
    int cur_dev_id = NpuDeviceHelper::GetCurrentDeivce();
    if (cur_dev_id != dev_id) {
      std::cout << "Current Device ID is " << cur_dev_id
                << ", but get Device id is " << dev_id
                << ".\t switch to device " << dev_id << std::endl;
      NpuDeviceHelper::SetCurrentDevice(dev_id);
      // _context.reset(new NpuContext(dev_id));
      // _context->SetCurrentContext();
      _prev_device.reset(new NpuDevice(cur_dev_id));
    }
    _stream.reset(new NpuStream());
  }

  ~NpuGuard() {
    if (_stream.get()) {
      _stream->Wait();
    }

    if (_prev_device.get()) {
      NpuDeviceHelper::SetCurrentDevice(_prev_device->GetId());
    }
  }

  aclrtStream GetStream() { return _stream->GetStream(); }

  void Wait() { _stream->Wait(); }

 private:
  std::shared_ptr<NpuStream> _stream{nullptr};
  std::shared_ptr<NpuContext> _context{nullptr};
  std::shared_ptr<NpuDevice> _prev_device{nullptr};
};
