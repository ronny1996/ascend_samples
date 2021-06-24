#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  {
    NpuTensor<float> x_tensor({2, 3, 2}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    NpuTensor<float> out_tensor({3});
    {
      NpuRunner runner("ReduceSumD");
      runner.AddInput(x_tensor).AddOutput(
          out_tensor)
          .SetAttr("axes", std::vector<int64_t>({0, 2}))
          .SetAttr("keep_dims", false)
          .Run();
    }
    out_tensor.print();
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
