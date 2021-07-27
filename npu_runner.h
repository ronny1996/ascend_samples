#pragma once
#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>
#include <type_traits>
#include <cstring>

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"

namespace npu {
  class alignas(2) float16 {
    public:
      uint16_t x;
      // The following defaulted special class member functions
      // are added to make float16 pass the std::is_trivial test
      float16() = default;
      float16(const float16& o) = default;
      float16& operator=(const float16& o) = default;
      float16(float16&& o) = default;
      float16& operator=(float16&& o) = default;
      ~float16() = default;
      explicit float16(const float& fp32) {
        // Conversion routine adapted from
        // http://stackoverflow.com/questions/1659440/32-bit-to-16-bit-floating-point-conversion
        float fp32_val = fp32;
        uint32_t fp32_bits = *((uint32_t*)(&fp32_val));
        x = ((fp32_bits >> 16) & 0x8000) | ((((fp32_bits >> 23) - 127 + 15) & 0x1f) << 10 )| ((fp32_bits >> 13) & 0x3ff);
      }

      explicit operator float() const {
        uint32_t fp32_bits = ((this->x & 0x8000) << 16) | (((((this->x >> 10) & 0x1f) - 15 + 127) & 0xff) << 23) | ((this->x & 0x03FF) << 13);
        return *((float*)(&fp32_bits));
      }
  };
};

#define ACL_CHECK(func)                                                      \
  do {                                                                       \
    auto status = func;                                                      \
    if (status != ACL_ERROR_NONE) {                                          \
      std::cerr << "call " << #func << " failed : " << status << " at file " \
                << __FILE__ << " line " << __LINE__ << std::endl;            \
      exit(-1);                                                              \
    }                                                                        \
  } while (0)

////////// AclDataType<T>::type
////////// AclDataType<T>::size()
template <typename T>
class AclDataType;
#define DECLAER_ACL_CPP_TYPE_MAP(acltype, cpptype)         \
  template <>                                              \
  struct AclDataType<cpptype> {                            \
    static const aclDataType type = acltype;               \
    static size_t size() { return aclDataTypeSize(type); } \
  }
DECLAER_ACL_CPP_TYPE_MAP(ACL_BOOL, bool);
DECLAER_ACL_CPP_TYPE_MAP(ACL_UINT8, uint8_t);
DECLAER_ACL_CPP_TYPE_MAP(ACL_UINT16, uint16_t);
DECLAER_ACL_CPP_TYPE_MAP(ACL_INT8, int8_t);
DECLAER_ACL_CPP_TYPE_MAP(ACL_INT8, char);
DECLAER_ACL_CPP_TYPE_MAP(ACL_INT16, int16_t);
DECLAER_ACL_CPP_TYPE_MAP(ACL_INT32, int32_t);
DECLAER_ACL_CPP_TYPE_MAP(ACL_INT64, int64_t);
DECLAER_ACL_CPP_TYPE_MAP(ACL_FLOAT16, npu::float16);
DECLAER_ACL_CPP_TYPE_MAP(ACL_FLOAT, float);
DECLAER_ACL_CPP_TYPE_MAP(ACL_DOUBLE, double);

////////// AclSetAttr(...)
#define DECLARE_ACL_SET_ATTR(cpptype, func)                     \
  void AclSetAttr(aclopAttr* attr, const std::string& attrname, \
                  const std::remove_const<cpptype>::type& t) {  \
    func(attr, attrname.c_str(), t);                            \
  }
DECLARE_ACL_SET_ATTR(bool, aclopSetAttrBool);
DECLARE_ACL_SET_ATTR(int64_t, aclopSetAttrInt);
DECLARE_ACL_SET_ATTR(float, aclopSetAttrFloat);
DECLARE_ACL_SET_ATTR(const char*, aclopSetAttrString);
#undef DECLARE_ACL_SET_ATTR
#define DECLARE_ACL_SET_ATTR(cpptype, func)                              \
  void AclSetAttr(aclopAttr* attr, const std::string& attrname, int num, \
                  const std::remove_const<cpptype>::type t) {            \
    func(attr, attrname.c_str(), num, t);                                \
  }
DECLARE_ACL_SET_ATTR(const uint8_t*, aclopSetAttrListBool);
DECLARE_ACL_SET_ATTR(const int64_t*, aclopSetAttrListInt);
DECLARE_ACL_SET_ATTR(const float*, aclopSetAttrListFloat);
DECLARE_ACL_SET_ATTR(const char**, aclopSetAttrListString);
#undef DECLARE_ACL_SET_ATTR
#define DECLARE_ACL_SET_ATTR(cpptype, func)                                  \
  void AclSetAttr(aclopAttr* attr, const std::string& attrname, int numlist, \
                  const int* numvalues,                                      \
                  const std::remove_const<cpptype>::type& t) {               \
    func(attr, attrname.c_str(), numlist, numvalues, t);                     \
  }
DECLARE_ACL_SET_ATTR(const int64_t* const*, aclopSetAttrListListInt);
#undef DECLARE_ACL_SET_ATTR

//////// NpuTensor
struct AclTensor {
  aclTensorDesc* desc;
  aclDataBuffer* buffer;
  aclFormat format;
  aclMemType memtype;
};

template <typename TensorType>
struct NpuTensor : public AclTensor {
  using T = typename std::remove_cv<TensorType>::type;

  NpuTensor() {
    format = ACL_FORMAT_UNDEFINED;
    desc = aclCreateTensorDesc(ACL_DT_UNDEFINED, 0, nullptr, ACL_FORMAT_UNDEFINED);
    buffer = aclCreateDataBuffer(nullptr, 0);
  }

  NpuTensor(const std::vector<int64_t>& dims_,
            aclFormat format_ = ACL_FORMAT_NCHW)
      : NpuTensor(dims_,
                  std::vector<T>(std::accumulate(dims_.begin(), dims_.end(), 1,
                                                 std::multiplies<int64_t>())),
                  format_) {}

  NpuTensor(const std::vector<int64_t>& dims_, const std::vector<T>& data_,
            aclFormat format_ = ACL_FORMAT_NCHW, aclMemType memtype_ = ACL_MEMTYPE_DEVICE)
      : host_data(data_), dims(dims_) {
    format = format_;
    memtype = memtype_;
    desc = aclCreateTensorDesc(AclDataType<T>::type, dims.size(), dims.data(),
                              format_);
    aclSetTensorStorageFormat(desc, format_);
    aclSetTensorStorageShape(desc, dims.size(), dims.data());
    dev_size = host_data.size() * AclDataType<T>::size();
    // like shape, dims ...
    bool is_host_tensor = (memtype == ACL_MEMTYPE_HOST) || is_const;
    if (is_host_tensor) {
      memtype = ACL_MEMTYPE_HOST;
      ACL_CHECK(aclSetTensorPlaceMent(desc, memtype));
      ACL_CHECK(aclrtMallocHost(&dev_ptr, dev_size));
      memcpy(dev_ptr, data_.data(), dev_size);
    } else { // ACL_MEMTYPE_DEVICE
      ACL_CHECK(aclrtMalloc(&dev_ptr, dev_size, ACL_MEM_MALLOC_HUGE_FIRST));
      ACL_CHECK(aclrtMemcpy(dev_ptr, dev_size, host_data.data(), dev_size,
                              ACL_MEMCPY_HOST_TO_DEVICE));
    }
    buffer = aclCreateDataBuffer(dev_ptr, dev_size);
    if (is_host_tensor) {
      ACL_CHECK(aclSetTensorConst(desc, dev_ptr, dev_size));
    }
  }

  ~NpuTensor() {
    // if (!is_const) return;
    if (format != ACL_FORMAT_UNDEFINED) {
      if (dev_ptr) {
        if (memtype != ACL_MEMTYPE_HOST) {
          ACL_CHECK(aclrtFree(dev_ptr));
        } else {
          ACL_CHECK(aclrtFreeHost(dev_ptr));
        }
      }
      aclDestroyTensorDesc(desc);
      aclDestroyDataBuffer(buffer);
    } else {
      // aclDestroyTensorDesc(desc);
    }
  }

  void sync() {
    if (format != ACL_FORMAT_UNDEFINED) {
      ACL_CHECK(aclrtMemcpy(host_data.data(), dev_size, dev_ptr, dev_size,
                            ACL_MEMCPY_DEVICE_TO_HOST));
    } else {
      std::cerr << "can't sync an undefined tesnor\n";
      exit(-1);
    }
  }

  void print() {
    sync();
    for (auto t : host_data) {
      std::cout << float(t) << ',';
    }
    std::cout << '\n';
  }

  std::vector<T> host_data;
  void* dev_ptr;
  size_t dev_size;
  std::vector<int64_t> dims;
  bool is_const = std::is_const<TensorType>::value;
};

struct NpuGuard {
  NpuGuard(int dev_id) {
    int cur_dev_id;
    ACL_CHECK(aclrtGetDevice(&cur_dev_id));
    if (cur_dev_id != dev_id) {
      std::cout << "Current Device ID is " << cur_dev_id
                << ", but get Device id is " << dev_id
                << ".\t switch to device " << dev_id << std::endl;
      ACL_CHECK(aclrtSetDevice(dev_id));
      // ACL_CHECK(aclrtCreateContext(&context, dev_id));
      // ACL_CHECK(aclrtSetCurrentContext(context));
      prev_dev_id = cur_dev_id;
    }
    ACL_CHECK(aclrtCreateStream(&stream));
  }

  ~NpuGuard() {
    if (stream) {
      ACL_CHECK(aclrtSynchronizeStream(stream));
      ACL_CHECK(aclrtDestroyStream(stream));
    }
    if (context) {
      ACL_CHECK(aclrtDestroyContext(context));
    }
    if (prev_dev_id != -1) {
      ACL_CHECK(aclrtSetDevice(prev_dev_id));
    }
  }
  aclrtContext context = nullptr;
  aclrtStream stream = nullptr;

private:
  int prev_dev_id = -1;
};

struct NpuRunner {
  NpuRunner(std::string optype_, int devid = 0) : optype(optype_), guard(0) {
    attr = aclopCreateAttr();
  }
  ~NpuRunner() { aclopDestroyAttr(attr); }
  NpuRunner& AddInput(AclTensor& tensor) {
    in_descs.push_back(tensor.desc);
    in_buffers.push_back(tensor.buffer);
    return *this;
  }
  NpuRunner& AddOutput(AclTensor& tensor) {
    out_descs.push_back(tensor.desc);
    out_buffers.push_back(tensor.buffer);
    return *this;
  }
  template <typename T>
  NpuRunner& SetAttr(const std::string& attrname, const T& t) {
    AclSetAttr(attr, attrname, t);
    return *this;
  }
  NpuRunner& SetAttr(const std::string& attrname, const int32_t& t) {
    AclSetAttr(attr, attrname, static_cast<int64_t>(t));
    return *this;
  }
  template <typename T>
  NpuRunner& SetAttr(const std::string& attrname, const std::vector<T>& t) {
    AclSetAttr(attr, attrname, t.size(), t.data());
    return *this;
  }
  NpuRunner& SetAttr(const std::string& attrname,
                     const std::vector<int32_t>& t) {
    std::vector<int64_t> tt;
    for (auto& v : t) {
      tt.push_back(static_cast<int64_t>(v));
    }
    AclSetAttr(attr, attrname, tt.size(), tt.data());
    return *this;
  }
  void Run() {
    ACL_CHECK(aclopCompileAndExecute(
        optype.c_str(), in_descs.size(), in_descs.data(), in_buffers.data(),
        out_descs.size(), out_descs.data(), out_buffers.data(), attr,
        ACL_ENGINE_SYS, ACL_COMPILE_SYS, NULL, guard.stream));
  }

  NpuGuard guard;
  std::vector<aclTensorDesc*> in_descs;
  std::vector<aclDataBuffer*> in_buffers;
  std::vector<aclTensorDesc*> out_descs;
  std::vector<aclDataBuffer*> out_buffers;
  aclopAttr* attr;
  std::string optype;
};

struct NpuHelper {
  static void SetDevice(int dev_id) {
    if (dev_id > NpuHelper::GetDevicesCount() || dev_id < 0) {
      std::cerr << "dev_id > NpuHelper::GetDevicesCount() || dev_id < 0" << std::endl;
    } else {
      aclrtSetDevice(dev_id);
    }
  }
  static void InitAllDevices() {
    ACL_CHECK(aclInit(nullptr));
    for (auto i = 0; i < NpuHelper::GetDevicesCount(); i++) {
      ACL_CHECK(aclrtSetDevice(i));
    }
    // aclrtSetExceptionInfoCallback(NpuHelper::ExceptionCallback);
  }
  static void ReleaseAllDevices() {
    for (auto i = 0; i < NpuHelper::GetDevicesCount(); i++) {
      ACL_CHECK(aclrtResetDevice(i));
    }
    ACL_CHECK(aclFinalize());
  }
  static uint32_t GetDevicesCount() {
    uint32_t count;
    ACL_CHECK(aclrtGetDeviceCount(&count));
    return count;
  }
  static void ExceptionCallback(aclrtExceptionInfo* exceptionInfo) {
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
    // print
  }
  static void GetRecentErrorMessage() {
      const char *aclRecentErrMsg = nullptr;
      aclRecentErrMsg = aclGetRecentErrMsg();
      if (aclRecentErrMsg != nullptr) {
        printf("%s\n", aclRecentErrMsg);
      } else {
        printf("Failed to get recent error message.\n");
      }
  }
};