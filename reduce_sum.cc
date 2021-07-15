#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  // for (auto i = 0; i < 50000; i++) {
  //   {
  //     NpuTensor<float> x_tensor({32, 102});
  //     NpuTensor<float> out_tensor({102});
  //     {
  //       NpuRunner runner("ReduceSumD");
  //       runner.AddInput(x_tensor).AddOutput(
  //           out_tensor)
  //           .SetAttr("axes", std::vector<int64_t>({0}))
  //           .SetAttr("keep_dims", false)
  //           .Run();
  //     }
  //     out_tensor.print();
  //   }
  //   std::cout << "Step " << i << std::endl;
  // }
  {
    NpuTensor<float> x_tensor({3, 4, 5});
    NpuTensor<float> out_tensor({1});
    {
      NpuRunner runner("ReduceMeanD");
      runner.AddInput(x_tensor)
      .AddOutput(out_tensor)
      .SetAttr("axes", std::vector<int64_t>({0, 1, 2}))
      .SetAttr("keep_dims", false)
      .Run();
    }
    out_tensor.print();
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
