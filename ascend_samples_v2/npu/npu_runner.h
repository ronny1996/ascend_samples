#pragma once
#include "npu_device.h"
#include "npu_internal_helper.h"
#include "npu_profiler.h"
#include "npu_tensor.h"

class NpuRunner {
 public:
  struct Builder {
    Builder(const std::string optype) : _optype(optype) {}

    Builder& AddInputs(std::vector<NpuTensorBase> tensors) {
      for (auto& tensor : tensors) {
        this->AddInput(tensor);
      }
      return *this;
    }

    Builder& AddOutputs(std::vector<NpuTensorBase> tensors) {
      for (auto& tensor : tensors) {
        this->AddOutput(tensor);
      }
      return *this;
    }

    Builder& AddInput(NpuTensorBase tensor) {
      this->inputs.push_back(tensor);
      return *this;
    }

    Builder& AddOutput(NpuTensorBase tensor) {
      this->outputs.push_back(tensor);
      return *this;
    }

    std::string _optype;
    std::vector<NpuTensorBase> inputs;
    std::vector<NpuTensorBase> outputs;
  };

  NpuRunner(Builder& builder) {
    this->Build(builder);
    this->attr = aclopCreateAttr();
    // ACL_CHECK(aclopCreateHandle(op_type.c_str(), in_descs.size(),
    //                             in_descs.data(), out_descs.size(),
    //                             out_descs.data(), this->attr, &this->_handle));
  }

  ~NpuRunner() {
    aclopDestroyAttr(this->attr);       // no check
    aclopDestroyHandle(this->_handle);  // no check
  }

  NpuRunner& Build(Builder& builder) {
    this->op_type = builder._optype;
    for (auto& tensor : builder.inputs) {
      this->in_descs.emplace_back(tensor.desc);
      this->in_buffers.emplace_back(tensor.buffer);
    }
    for (auto& tensor : builder.outputs) {
      this->out_descs.emplace_back(tensor.desc);
      this->out_buffers.emplace_back(tensor.buffer);
    }
    return *this;
  }

  void Run(aclrtStream stream) {
    ACL_CHECK(aclopCompileAndExecute(
        op_type.c_str(), in_descs.size(), in_descs.data(), in_buffers.data(),
        out_descs.size(), out_descs.data(), out_buffers.data(), attr,
        ACL_ENGINE_SYS, ACL_COMPILE_SYS, NULL, stream));
  }

  void Execute(aclrtStream stream) {
    ACL_CHECK(aclopExecuteV2(
        op_type.c_str(), in_descs.size(), in_descs.data(), in_buffers.data(),
        out_descs.size(), out_descs.data(), out_buffers.data(), attr, stream));
  }

  void RunWithHandle(aclrtStream stream) {
    ACL_CHECK(aclopExecWithHandle(_handle, in_descs.size(), in_buffers.data(),
                                  out_descs.size(), out_buffers.data(),
                                  stream));
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

  template <typename... ARGS, int N = sizeof...(ARGS)>
  NpuRunner& SetAttrs(std::pair<std::string, ARGS>... attrs) {
    return *this;
  }

 private:
  std::string op_type;
  std::vector<aclTensorDesc*> in_descs;
  std::vector<aclDataBuffer*> in_buffers;
  std::vector<aclTensorDesc*> out_descs;
  std::vector<aclDataBuffer*> out_buffers;
  aclopAttr* attr = nullptr;

  aclopHandle* _handle = nullptr;

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
};
