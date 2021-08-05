#pragma once
#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>
#include <type_traits>
#include <cstring>
#include <cassert>

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"

#define ACL_CHECK(func)                                                      \
  do {                                                                       \
    auto status = func;                                                      \
    if (status != ACL_ERROR_NONE) {                                          \
      std::cerr << "call " << #func << " failed : " << status << " at file " \
                << __FILE__ << " line " << __LINE__ << std::endl;            \
      {                                                                      \
      const char *aclRecentErrMsg = nullptr;                                 \
      aclRecentErrMsg = aclGetRecentErrMsg();                                \
      if (aclRecentErrMsg != nullptr) {                                      \
        printf("\n%s\n", aclRecentErrMsg);                                   \
      } else {                                                               \
        printf("Failed to get recent error message.\n");                     \
      }                                                                      \
      }                                                                      \
      exit(-1);                                                              \
    }                                                                        \
  } while (0)



class NpuRunner {
  public:
    class Builder {
      std::string op_type;
      public:
        Builder(const std::string _op_type) : op_type(_op_type) {

        }
        const std::string& GetOpType() {
          return op_type;
        }
    }
    
    NpuRunner(Builder &builder) {
      this->op_type = builder.GetOpType();
    }

    void Run() {
      aclopExecuteV2
    }
  private:
    std::string op_type;
}