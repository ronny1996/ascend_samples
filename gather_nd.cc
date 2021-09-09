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
    std::vector<float> x_data;
    for (auto i = 0; i < 2 * 3 * 4; i++) {
      x_data.push_back(i);
    }
    NpuTensor<float> x_tensor({2, 3, 4}, x_data);
    NpuTensor<int64_t> index_tensor({1, 2}, {0, 2});
    NpuTensor<float> out_tensor({4});
    {
      NpuRunner runner("GatherV2D");
      runner.AddInput(x_tensor)
          .AddInput(index_tensor)
          .AddOutput(out_tensor)
          .SetAttr("axis", 0)
          .Run();
    }
    out_tensor.print();

    // gather_elements_op(x_tensor, index_tensor);
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
