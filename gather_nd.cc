#include "npu_runner.h"

auto reshape_op = [](AclTensor& in, AclTensor& out, const std::vector<int32_t> &new_shape) {
  NpuRunner runner("Reshape");
  NpuTensor<const int32_t> shape({new_shape.size()}, new_shape);
  runner.AddInput(in).AddInput(shape)
  .AddOutput(out)
  .SetAttr("axis", 0)
  .Run();
};

auto gather_elements_op = [](NpuTensor<float>& in, NpuTensor<int64_t>& index, int dim = 0) {
  auto &in_dims = in.dims;
  auto &index_dims = index.dims;
  int32_t in_numel = static_cast<int32_t>(CommonHelper::Product(in_dims));
  int32_t index_numel = static_cast<int32_t>(CommonHelper::Product(index_dims));

  NpuTensor<float> in_reshape;
  NpuTensor<int64_t> index_reshape;
  reshape_op(in, in_reshape, {in_numel});
  reshape_op(index, index_reshape, {index_numel});
  in_reshape.print();
  index_reshape.print();
};


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    NpuTensor<float> x_tensor({8}, {1, 2, 3, -4, 5, -6, 7, -8});
    NpuTensor<int64_t> index_tensor({2}, {1, 0});
    NpuTensor<float> out_tensor({2});
    {
      NpuRunner runner("GatherV2D");
      runner.AddInput(x_tensor)
          .AddInput(index_tensor)
          .AddOutput(out_tensor)
          .SetAttr("axis", 0)
          .Run();
    }
    out_tensor.print();

    gather_elements_op(x_tensor, index_tensor);
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
