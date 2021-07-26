#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    NpuTensor<float> x_tensor({2, 3}, {-1, -2, -3, -4, -5, -6});
    NpuTensor<const int32_t> num_samples({1}, {2}, ACL_FORMAT_NCHW, ACL_MEMTYPE_HOST);
    NpuTensor<const int32_t> seed({2}, {1, 2}, ACL_FORMAT_NCHW, ACL_MEMTYPE_HOST);
    NpuTensor<int32_t> out_tensor({2, 2});
    {
      NpuRunner runner("Multinomial");
      runner.AddInput(x_tensor)
          .AddInput(num_samples)
          // .AddInput(seed)
          .AddOutput(out_tensor)
          .SetAttr("output_dtype", static_cast<int64_t>(ACL_INT32))
          .SetAttr("seed", static_cast<int64_t>(1))
          .SetAttr("seed2", static_cast<int64_t>(2))
          .Run();
    }
    out_tensor.print();
  }
  {
    NpuTensor<float> x_tensor({6}, {0.2, 0.3, 0.5, 0.7, 0.8, 0.9});
    NpuTensor<int32_t> out_tensor({2});
    {
      NpuRunner runner("MultinomialWithReplacementD");
      runner.AddInput(x_tensor)
          .AddOutput(out_tensor)
          .SetAttr("num_samples", 2)
          .SetAttr("replacement", true)
          .Run();
    }
    out_tensor.print();
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
