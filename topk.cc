#include "npu_runner.h"


template<typename T>
std::vector<T> gen_assist_seq(int64_t dim) {
  const int64_t dimx2 = dim;
  std::vector<T> assit;
  assit.resize(2 * dimx2);
  for (int64_t i = 0; i < dimx2; i++) {
    // for i in range [0, dim]
    assit[i] = static_cast<T>(i);
    // for i in range [dim, dimx2]
    int64_t idx =
        static_cast<int64_t>(static_cast<float>(static_cast<T>(i)));
    int64_t gap = i - idx;
    assit[i + dim] = static_cast<T>(gap);
  }
  return assit;
}


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    int32_t k = 2;
    NpuTensor<float> x_tensor({3, 3}, std::initializer_list<float>({0.78104149, 0.88745828, 0.32362268, 0.82196718, 0.48763277, 0.42826136, 0.96527182, 0.34851612, 0.12959783}));
    NpuTensor<int32_t> k_tensor({1}, {k});
    NpuTensor<float> out_tensor({3, k});
    NpuTensor<int32_t> indices_tesnor({3, k});
    NpuTensor<float> assist_seq({4}, gen_assist_seq<float>(2));
    #if 0
    {
      NpuRunner runner("TopK");
      runner.AddInput(x_tensor)
          .AddInput(k_tensor)
          .AddOutput(out_tensor)
          .AddOutput(indices_tesnor)
          .SetAttr("sorted", true) // must be false
          .SetAttr("largest", true)
          .SetAttr("dim", -1)
          .Run();
    }
    #else
    {
      NpuRunner runner("TopKD");
      runner.AddInput(x_tensor)
          .AddInput(assist_seq)
          .AddOutput(out_tensor)
          .AddOutput(indices_tesnor)
          .SetAttr("k", k)
          .SetAttr("sorted", true) // must be false
          .SetAttr("largest", true)
          .SetAttr("dim", -1)
          .Run();
    }
    #endif
    x_tensor.print();
    out_tensor.print();
    indices_tesnor.print();
    NpuTensor<int64_t> indices({3, k});
    {
      NpuRunner runner("Cast");
      runner.AddInput(indices_tesnor).AddOutput(indices).SetAttr("dst_type", static_cast<int32_t>(AclDataType<int64_t>::type)).Run();
    }
    indices.print();

    NpuTensor<float> out({12, k});
    {
      NpuRunner runner("GatherV2D");
      runner.AddInput(x_tensor)
          .AddInput(indices_tesnor)
          .AddOutput(out)
          .SetAttr("axis", 1)
          .Run();
    }
    out.print();
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
