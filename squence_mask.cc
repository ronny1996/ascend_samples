#include "npu_runner.h"



template <typename Tx, typename Ty>
void squence_mask(NpuTensor<Tx> &lengths, NpuTensor<Ty> &mask, int32_t maxlen, int devid) {
  NpuTensor<int32_t> cast_lengths(lengths.dims);
  {
    NpuRunner runner("Cast");
    runner.AddInput(lengths)
        .AddOutput(cast_lengths)
        .SetAttr("dst_type", static_cast<int32_t>(AclDataType<int32_t>::type))
        .Run();
  }
  auto cast_lengths_dims = lengths.dims;
  cast_lengths_dims.push_back(1);
  cast_lengths.sync();
  NpuTensor<int32_t> cast_lengths_with_new_dim(cast_lengths_dims, cast_lengths.host_data);

  NpuTensor<int32_t> range_tensor({maxlen});
  NpuTensor<const int32_t> start({1}, {0});
  NpuTensor<const int32_t> limit({1}, {maxlen});
  NpuTensor<const int32_t> delta({1}, {1});
  {
    NpuRunner runner("Range");
    runner.AddInput(start)
        .AddInput(limit)
        .AddInput(delta)
        .AddOutput(range_tensor)
        .Run();
  }
  range_tensor.print();

  auto expand_dims = lengths.dims;
  expand_dims.push_back(maxlen);
  NpuTensor<int32_t> mask_tensor(expand_dims);
  {
    NpuRunner runner("ExpandD");
    runner.AddInput(range_tensor)
        .AddOutput(mask_tensor)
        .SetAttr("shape", expand_dims)
        .Run();
  }
  mask_tensor.print();

  NpuTensor<int32_t> lengths_tensor(expand_dims);
  {
    NpuRunner runner("TileWithAxis");
    runner.AddInput(cast_lengths_with_new_dim)
        .AddOutput(lengths_tensor)
        .SetAttr("axis", static_cast<int64_t>(lengths.dims.size()))
        .SetAttr("tiles", static_cast<int64_t>(maxlen))
        .Run();
  }
  lengths_tensor.print();

  NpuTensor<uint8_t> out_tensor(expand_dims);
  {
    NpuRunner runner("Less");
    runner.AddInput(mask_tensor)
        .AddInput(lengths_tensor)
        .AddOutput(out_tensor)
        .Run();
  }
  out_tensor.print();

  // NpuTensor<T> mask(expand_dims);
  {
    NpuRunner runner("Cast");
    runner.AddInput(out_tensor)
        .AddOutput(mask)
        .SetAttr("dst_type", static_cast<int32_t>(AclDataType<Ty>::type))
        .Run();
  }
  mask.print();
}

int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    int max_length = 10;
    std::vector<int64_t> dims({2, 3});
    NpuTensor<int64_t> lengths(dims, {0,3,4,5,7,8});
    dims.push_back(max_length);
    NpuTensor<int32_t> mask(dims);
    squence_mask(lengths, mask, 10, 0);
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
