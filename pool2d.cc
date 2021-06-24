#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  {
    NpuTensor<float> x_tensor({1, 1, 2, 2}, {1, 5, 2, 2});

    NpuTensor<float> out_tensor({1, 1, 1, 1});
    {
      NpuRunner runner("Pooling");
      runner.AddInput(x_tensor)
          .AddOutput(out_tensor)
          .SetAttr("mode", static_cast<int64_t>(0))
          .SetAttr("global_pooling", false)
          .SetAttr("window", std::vector<int64_t>({2, 2}))
          .SetAttr("stride", std::vector<int64_t>({1, 1}))
          .SetAttr("pad", std::vector<int64_t>({0, 0, 0, 0}))
          .SetAttr("dilation", std::vector<int64_t>({1, 1, 1, 1}))
          .SetAttr("ceil_mode", static_cast<int64_t>(0))
          .Run();
    }
    out_tensor.print();
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
