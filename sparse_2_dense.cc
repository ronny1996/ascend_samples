#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    NpuTensor<int32_t> indices({2, 2}, {0, 2, 1, 0});
    NpuTensor<int32_t> output_shape({2}, {2, 4});
    NpuTensor<float> values({2}, {1, 1});
    NpuTensor<float> default_values({1}, {0});
    NpuTensor<float> out_tensor({2, 4});
    {
      NpuRunner runner("SparseToDense");
      /**
       *•indices: A 0D, 1D, or 2D Tensor of type int32 or int64. 
        •output_shape: A 1D Tensor of the same type as "sparse_indices". The shape of the dense output tensor. 
        •values: A 1D Tensor. Values corresponding to each row of "sparse_indices", or a scalar value to be used for all sparse indices. 
        •default_value: A Tensor of the same type as "sparse_values" . 
        */
      runner.AddInput(indices)
          .AddInput(output_shape)
          .AddInput(values)
          .AddInput(default_values)
          .AddOutput(out_tensor)
          .Run();
    }
    out_tensor.print();
  }
  
  NpuHelper::ReleaseAllDevices();
  return 0;
}
