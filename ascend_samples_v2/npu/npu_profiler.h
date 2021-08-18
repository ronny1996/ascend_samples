#pragma once
#include "npu_device.h"

struct Profiler {
  // For CANN 20.2+
  // ACL_AICORE_ARITHMETIC_UTILIZATION = 0, record arithmetic stats
  // ACL_AICORE_PIPE_UTILIZATION = 1, record pipeline
  // ACL_AICORE_MEMORY_BANDWIDTH = 2, record memory
  // ACL_AICORE_L0B_AND_WIDTH = 3, recore internal memory
  // ACL_AICORE_RESOURCE_CONFLICT_RATIO = 5, record pipeline ratio
  static constexpr aclprofAicoreMetrics default_metrics =
      ACL_AICORE_ARITHMETIC_UTILIZATION;

  // ACL_PROF_ACL_API, record ACL API stats
  // ACL_PROF_TASK_TIME, record AI core stats
  // ACL_PROF_AICORE_METRICS, must include
  // ACL_PROF_AICPU_TRACE, recore AICPU, not supported yet
  static constexpr uint64_t default_type =
      ACL_PROF_ACL_API | ACL_PROF_AICORE_METRICS | ACL_PROF_TASK_TIME;

  Profiler(const std::string& output_dir, std::vector<uint32_t> devices = {},
            aclprofAicoreMetrics metrics = Profiler::default_metrics,
            aclprofAicoreEvents* events = nullptr,
            uint64_t type = Profiler::default_type)
      : _output_dir(output_dir) {
    if (devices.empty()) {
      auto devices_count = NpuDeviceHelper::GetDevicesCount();
      for (auto i = 0; i < devices_count; i++) {
        devices.push_back(i);
      }
    }
    _config = aclprofCreateConfig(devices.data(), devices.size(), metrics,
                                  events, type);
    if (!_config) {
      std::cerr << "Failed to call aclprofCreateConfig" << std::endl;
    } else {
      std::cout << "Start profiling...\n";
      ACL_CHECK(aclprofInit(output_dir.c_str(), output_dir.size()));
      ACL_CHECK(aclprofStart(_config));
    }
  }

  ~Profiler() {
    if (_config) {
      std::cout << "Finish profiling collection.\n";
      ACL_CHECK(aclprofStop(_config));
      ACL_CHECK(aclprofDestroyConfig(_config));
      ACL_CHECK(aclprofFinalize());
    }
  }

  aclprofConfig* _config = nullptr;
  std::string _output_dir;
};