#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  // NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
<<<<<<< Updated upstream
#if 1 
=======
  #if 1
>>>>>>> Stashed changes
  {
    std::vector<float> x_data;
    for(auto i = 0; i < 200; i++) {
      x_data.push_back(i + 1);
    }
    std::vector<int64_t> idx_data;
    for(auto i = 0; i < 100; i++) {
      idx_data.push_back(static_cast<int64_t>(rand() % 20));
      std::cout << idx_data.back() << std::endl;
    }
    NpuTensor<float> x_tensor({10, 20}, x_data);
    NpuTensor<int64_t> index_tensor({10, 10}, idx_data);
    NpuTensor<float> out_tensor({10, 10});

    {
      NpuRunner runner("GatherElements");
      runner.AddInput(x_tensor)
          .AddInput(index_tensor)
          // .AddInput(axis)
          .AddOutput(out_tensor)
          .SetAttr("dim",1)
          .Run();
    }
    out_tensor.print();
  }
<<<<<<< Updated upstream
#endif
=======
  #endif
>>>>>>> Stashed changes
  // 1 2
  // 3 -4

  // 5 -6
  // 7 -8
  #if 0
  {
    NpuTensor<float> x_tensor({2, 2, 2}, {1, 2, 3, -4, 5, -6, 7, -8});
    NpuTensor<int64_t> index_tensor({4, 3}, {0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1});
    NpuTensor<float> out_tensor({4});

    {
      NpuRunner runner("GatherNd");
      runner.AddInput(x_tensor)
          .AddInput(index_tensor)
          .AddOutput(out_tensor)
          .Run();
    }
    out_tensor.print();
  }
  #endif
  // NpuHelper::ReleaseAllDevices();
  return 0;
}
