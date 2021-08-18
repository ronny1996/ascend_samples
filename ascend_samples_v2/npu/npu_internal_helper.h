#pragma once
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "acl/acl_prof.h"

/**
 * @brief npu_internal_helper
 *
 * @macro ACL_CHECK
 *
 * @class DTypeFormatUtils
 *    @function std::string ToString(aclDataType dtype)
 *    @function std::string ToString(aclFormat format)
 *    @function aclDataType ToDtype(const std::string& str)
 *    @function aclFormat ToFormat(const std::string& str)
 *    @meta function CTypeToDtype<T> - type, size(), str()
 *
 * @class NpuDescInternalHelper
 */

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
    x = ((fp32_bits >> 16) & 0x8000) |
        ((((fp32_bits >> 23) - 127 + 15) & 0x1f) << 10) |
        ((fp32_bits >> 13) & 0x3ff);
  }

  explicit operator float() const {
    uint32_t fp32_bits =
        ((this->x & 0x8000) << 16) |
        (((((this->x >> 10) & 0x1f) - 15 + 127) & 0xff) << 23) |
        ((this->x & 0x03FF) << 13);
    return *((float*)(&fp32_bits));
  }
};

};  // namespace npu

#define ACL_CHECK(func)                                                      \
  do {                                                                       \
    auto status = func;                                                      \
    if (status != ACL_ERROR_NONE) {                                          \
      std::cerr << "call " << #func << " failed : " << status << " at file " \
                << __FILE__ << " line " << __LINE__ << std::endl;            \
      {                                                                      \
        const char* aclRecentErrMsg = nullptr;                               \
        aclRecentErrMsg = aclGetRecentErrMsg();                              \
        if (aclRecentErrMsg != nullptr) {                                    \
          printf("\n%s\n", aclRecentErrMsg);                                 \
        } else {                                                             \
          printf("Failed to get recent error message.\n");                   \
        }                                                                    \
      }                                                                      \
      exit(-1);                                                              \
    }                                                                        \
  } while (0)

std::vector<std::pair<std::string, aclFormat>> string_format_mapping = {
    {"UNDEFINED", ACL_FORMAT_UNDEFINED},
    {"NCHW", ACL_FORMAT_NCHW},
    {"NHWC", ACL_FORMAT_NHWC},
    {"ND", ACL_FORMAT_ND},
    {"NC1HWC0", ACL_FORMAT_NC1HWC0},
    {"FRACTAL_Z", ACL_FORMAT_FRACTAL_Z},
    {"FRACTAL_NZ", ACL_FORMAT_FRACTAL_NZ},
};

std::vector<std::pair<std::string, aclDataType>> string_dtype_mapping = {
    {"UNDEFINED", ACL_DT_UNDEFINED},
    {"FP16", ACL_FLOAT16},
    {"FP32", ACL_FLOAT},
    {"FP64", ACL_DOUBLE},
    {"INT8", ACL_INT8},
    {"INT16", ACL_INT16},
    {"INT32", ACL_INT32},
    {"INT64", ACL_INT64},
    {"UINT8", ACL_UINT8},
    {"UINT16", ACL_UINT16},
    {"UINT32", ACL_UINT32},
    {"UINT32", ACL_UINT64},
    {"BOOL", ACL_BOOL},
};

std::vector<std::pair<std::string, aclrtRunMode>> string_run_mode_mapping = {
    {"DEVICE", ACL_DEVICE},
    {"HOST", ACL_HOST},
};

struct DTypeFormatUtils {
  static std::string ToString(aclDataType dtype) {
    for (const auto& pair : string_dtype_mapping) {
      if (pair.second == dtype) {
        return pair.first;
      }
    }
    std::cerr << "[TypeToString] cant find " << static_cast<int>(dtype)
              << std::endl;
    exit(0);
  }

  static aclDataType ToDtype(const std::string& str) {
    for (const auto& pair : string_dtype_mapping) {
      if (pair.first == str) {
        return pair.second;
      }
    }
    std::cerr << "[StringToType] cant find " << str << std::endl;
    exit(0);
  }

  static std::string ToString(aclFormat format) {
    for (const auto& pair : string_format_mapping) {
      if (pair.second == format) {
        return pair.first;
      }
    }
    std::cerr << "[FormatToString] cant find " << static_cast<int>(format)
              << std::endl;
    exit(0);
  }

  static aclFormat ToFormat(const std::string& str) {
    for (const auto& pair : string_format_mapping) {
      if (pair.first == str) {
        return pair.second;
      }
    }
    std::cerr << "[StringToFormat] cant find " << str << std::endl;
    exit(0);
  }

  template <typename T, typename Dummy = void>
  struct CTypeToDtype;
#define DECLARE_MAPPING(acltype, cpptype)                  \
  template <typename Dummy>                                \
  struct CTypeToDtype<cpptype, Dummy> {                    \
    static const aclDataType type = acltype;               \
    static size_t size() { return aclDataTypeSize(type); } \
    static std::string str() { return ToString(type); }    \
  }
  DECLARE_MAPPING(ACL_DT_UNDEFINED, void);
  DECLARE_MAPPING(ACL_BOOL, bool);
  DECLARE_MAPPING(ACL_UINT32, uint32_t);
  DECLARE_MAPPING(ACL_UINT16, uint16_t);
  DECLARE_MAPPING(ACL_UINT8, uint8_t);
  DECLARE_MAPPING(ACL_UINT64, uint64_t);
  DECLARE_MAPPING(ACL_INT8, int8_t);
  DECLARE_MAPPING(ACL_INT8, char);
  DECLARE_MAPPING(ACL_INT16, int16_t);
  DECLARE_MAPPING(ACL_INT32, int32_t);
  DECLARE_MAPPING(ACL_INT64, int64_t);
  DECLARE_MAPPING(ACL_FLOAT16, npu::float16);
  DECLARE_MAPPING(ACL_FLOAT, float);
  DECLARE_MAPPING(ACL_DOUBLE, double);
#undef DECLARE_MAPPING
};

struct NpuDescInternalHelper {
  static aclTensorDesc* TransFormatAndGenerateDesc(const aclTensorDesc* srcDesc,
                                                   aclFormat dstFormat) {
    aclTensorDesc* dstDesc;
    ACL_CHECK(aclTransTensorDescFormat(srcDesc, dstFormat, &dstDesc));
    return dstDesc;
  }

  // Setter for desc
  static void SetStorageShape(aclTensorDesc* desc, std::vector<int64_t> dims) {
    ACL_CHECK(aclSetTensorStorageShape(desc, dims.size(), dims.data()));
  }

  static void SetOriginShape(aclTensorDesc* desc, std::vector<int64_t> dims) {
    ACL_CHECK(aclSetTensorOriginShape(desc, dims.size(), dims.data()));
  }

  static void SetShape(aclTensorDesc* desc, std::vector<int64_t> dims) {
    ACL_CHECK(aclSetTensorShape(desc, dims.size(), dims.data()));
  }

  static void SetStorageFormat(aclTensorDesc* desc, aclFormat format) {
    ACL_CHECK(aclSetTensorStorageFormat(desc, format));
  }

  static void SetOriginFormat(aclTensorDesc* desc, aclFormat format) {
    ACL_CHECK(aclSetTensorOriginFormat(desc, format));
  }

  static void SetFormat(aclTensorDesc* desc, aclFormat format) {
    ACL_CHECK(aclSetTensorFormat(desc, format));
  }

  static void SetName(aclTensorDesc* desc, const std::string& name) {
    aclSetTensorDescName(desc, name.data());
  }

  // Getter for desc
  static aclFormat GetFormat(aclTensorDesc* desc) {
    auto format = aclGetTensorDescFormat(desc);
    return format;
  }

  static aclDataType GetDType(aclTensorDesc* desc) {
    auto dtype = aclGetTensorDescType(desc);
    return dtype;
  }

  static size_t GetMemorySize(aclTensorDesc* desc) {
    return aclGetTensorDescSize(desc);
  }

  static size_t GetElementCount(aclTensorDesc* desc) {
    return aclGetTensorDescElementCount(desc);
  }

  static std::vector<int64_t> GetDims(aclTensorDesc* desc) {
    auto n_dims = aclGetTensorDescNumDims(desc);
    std::vector<int64_t> dims(n_dims);
    for (size_t index = 0; index < n_dims; index++) {
      ACL_CHECK(aclGetTensorDescDimV2(desc, index, &dims[index]));
    }
    return dims;
  }

  static std::string GetName(aclTensorDesc* desc) {
    return aclGetTensorDescName(desc);
  }
};
