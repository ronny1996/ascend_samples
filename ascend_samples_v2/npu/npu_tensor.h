#pragma once
#include "npu_internal_helper.h"


struct NpuTensorBase {
  aclTensorDesc* desc = nullptr;
  aclDataBuffer* buffer = nullptr;

 NpuTensorBase& SetStorageShape(std::vector<int64_t> dims) {
    NpuDescInternalHelper::SetStorageShape(this->desc, dims);
    return *this;
  }

  NpuTensorBase& SetOriginShape(std::vector<int64_t> dims) {
    NpuDescInternalHelper::SetOriginShape(this->desc, dims);
    return *this;
  }

  NpuTensorBase& SetShape(std::vector<int64_t> dims) {
    NpuDescInternalHelper::SetShape(this->desc, dims);
    return *this;
  }

  NpuTensorBase& SetStorageFormat(const std::string& format) {
    NpuDescInternalHelper::SetStorageFormat(this->desc, DTypeFormatUtils::ToFormat(format));
    return *this;
  }

  NpuTensorBase& SetOriginFormat(const std::string& format) {
    NpuDescInternalHelper::SetOriginFormat(this->desc, DTypeFormatUtils::ToFormat(format));
    return *this;
  }

  NpuTensorBase& SetFormat(const std::string& format) {
    NpuDescInternalHelper::SetFormat(this->desc, DTypeFormatUtils::ToFormat(format));
    return *this;
  }

  NpuTensorBase& SetName(const std::string& name) {
    NpuDescInternalHelper::SetName(this->desc, name);
    return *this;
  }
};


template<typename TensorType>
struct NpuTensor : public NpuTensorBase {
  using T = typename std::remove_cv<TensorType>::type;

  NpuTensor() {
    format = ACL_FORMAT_UNDEFINED;
    desc =
        aclCreateTensorDesc(ACL_DT_UNDEFINED, 0, nullptr, ACL_FORMAT_UNDEFINED);
    buffer = aclCreateDataBuffer(nullptr, 0);
  }

  NpuTensor(const std::vector<int64_t>& dims_,
            aclFormat format_ = ACL_FORMAT_NCHW)
      : NpuTensor(dims_,
                  std::vector<T>(std::accumulate(dims_.begin(), dims_.end(), 1,
                                                 std::multiplies<int64_t>())),
                  format_) {}

  NpuTensor(const std::vector<int64_t>& dims_, const std::vector<T>& data_,
            aclFormat format_ = ACL_FORMAT_NCHW,
            aclMemType memtype_ = ACL_MEMTYPE_DEVICE)
      : host_data(data_), dims(dims_) {
    format = format_;
    memtype = memtype_;
    desc = aclCreateTensorDesc(DTypeFormatUtils::CTypeToDtype<T>::type, dims.size(), dims.data(),
                               format_);
    aclSetTensorStorageFormat(desc, format_);
    aclSetTensorStorageShape(desc, dims.size(), dims.data());
    dev_size = host_data.size() * DTypeFormatUtils::CTypeToDtype<T>::size();
    // like shape, dims ...
    bool is_host_tensor = (memtype == ACL_MEMTYPE_HOST) || is_const;
    if (is_host_tensor) {
      memtype = ACL_MEMTYPE_HOST;
      ACL_CHECK(aclSetTensorPlaceMent(desc, memtype));
      ACL_CHECK(aclrtMallocHost(&dev_ptr, dev_size));
      memcpy(dev_ptr, data_.data(), dev_size);
    } else {  // ACL_MEMTYPE_DEVICE
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
      // desc = nullptr;
      // buffer = nullptr;
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

  aclFormat format;
  aclMemType memtype;
};
