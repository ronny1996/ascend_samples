#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  // NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  // 1 2
  // 3 -4

  // 5 -6
  // 7 -8
  {
    NpuTensor<float> x_tensor({4}, {1, 3, 5, -8});
    NpuTensor<int32_t> index_tensor({4, 3}, {0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1});
    NpuTensor<float> out_tensor({2, 2, 2});
    NpuTensor<const int32_t> out_shape_tensor({3}, {2, 2, 2}, ACL_FORMAT_NCHW, ACL_MEMTYPE_HOST); 
    {
      NpuRunner runner("ScatterNd");
      runner.AddInput(index_tensor).AddInput(x_tensor)
          .AddInput(out_shape_tensor)
          .AddOutput(out_tensor)
          .Run();
    }
    out_tensor.print();
    // dont need clear
    NpuTensor<int32_t> index_tensor2({4, 3}, {0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0});
    {
      NpuRunner runner("ScatterNd");
      runner.AddInput(index_tensor2).AddInput(x_tensor)
          .AddInput(out_shape_tensor)
          .AddOutput(out_tensor)
          .Run();
    }
    out_tensor.print();
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
