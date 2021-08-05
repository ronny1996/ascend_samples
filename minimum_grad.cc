#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    NpuTensor<float> x1_tensor({2, 4}, {1, 2, 3, -4, 5, -6, 7, -8});
    NpuTensor<float> x2_tensor({2, 4}, {3, 1, 5, -6, 7, -1, 2, -4});
    NpuTensor<float> y1_tensor({2, 4});
    NpuTensor<float> y2_tensor({2, 4});
    NpuTensor<float> grad({2, 4}, std::vector<float>(2 * 4, 1.0f));
    {
      NpuRunner runner("MinimumGrad");
      runner.AddInput(grad)
          .AddInput(x1_tensor)
          .AddInput(x2_tensor)
          .AddOutput(y1_tensor)
          .AddOutput(y2_tensor)
          .SetAttr("grad_x", true)
          .SetAttr("grad_y", true)
          .Run();
    }
    y1_tensor.print(); // 1,0,1,0,1,1,0,1
    y2_tensor.print(); // 0,1,0,1,0,0,1,0
  }
  
  NpuHelper::ReleaseAllDevices();
  return 0;
}
