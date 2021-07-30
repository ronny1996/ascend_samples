#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  #if 0
  for (auto i = 0; i < 50000; i++) {
    {
      NpuTensor<float> x_tensor({32, 102});
      NpuTensor<float> out_tensor({102});
      {
        NpuRunner runner("ReduceSumD");
        runner.AddInput(x_tensor).AddOutput(
            out_tensor)
            .SetAttr("axes", std::vector<int64_t>({0}))
            .SetAttr("keep_dims", false)
            .Run();
      }
      out_tensor.print();
    }
    std::cout << "Step " << i << std::endl;
  }
  #endif
  #if 0
  {
    NpuTensor<float> x_tensor({2, 2}, {1, 2, 3, 4});
    NpuTensor<float> out_tensor({2});
    {
      NpuRunner runner("ReduceSumD");
      runner.AddInput(x_tensor)
      .AddOutput(out_tensor)
      .SetAttr("axes", std::vector<int64_t>({0}))
      .SetAttr("keep_dims", false)
      .Run();
    }
    out_tensor.print();
    {
      NpuRunner runner("ReduceSumD");
      runner.AddInput(x_tensor)
      .AddOutput(out_tensor)
      .SetAttr("axes", std::vector<int64_t>({0}))
      .SetAttr("keep_dims", false)
      .Run();
    }
    out_tensor.print();
  }
  #endif
  NpuHelper::ReleaseAllDevices();
  return 0;
}
