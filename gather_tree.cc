#include "npu_runner.h"


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  {
    {
      NpuTensor<int32_t> ids({3, 2, 2}, {2, 2, 6, 1, 3, 9, 6, 1, 0, 1, 9, 0});
      NpuTensor<int32_t> parent({3, 2, 2}, {0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1});
      NpuTensor<int32_t> out_tensor({3, 2, 2});
      // auto split_op = [] (AclTensor& in, AclTensor& out1, AclTensor& out2) {
      //   NpuRunner runner("SplitD");
      //   runner.AddInput(in).AddOutput(out1).AddOutput(out1).SetAttr("split_dim", 0).SetAttr("num_split", 2);
      // };

      auto slice_op = [](AclTensor& in, AclTensor& out, const std::vector<int32_t> &offsets, const std::vector<int32_t> &size) {
        NpuRunner runner("SliceD");
        runner.AddInput(in).AddOutput(out).SetAttr("offsets", offsets).SetAttr("size", size).Run();
      };

      NpuTensor<int32_t> ids_0_2({2, 2, 2}), ids_2_1({1, 2, 2});
      slice_op(ids, ids_0_2, {0, 0, 0}, {2, 2, 2});
      slice_op(ids, ids_2_1, {2, 0, 0}, {1, 2, 2});
      ids_0_2.print();
      ids_2_1.print();

      NpuTensor<int32_t> parent_1_2({2, 2, 2});
      slice_op(parent, parent_1_2, {1, 0, 0}, {2, 2, 2});
      parent_1_2.print();

      auto gether_elements_op = [](AclTensor& in,  AclTensor& index, const std::vector<int64_t> &index_dims, AclTensor& out) {
        NpuTensor<int64_t> index_int64(index_dims);
        NpuRunner cast_runner("Cast");
        cast_runner.AddInput(index).AddOutput(index_int64).SetAttr("dst_type", static_cast<int64_t>(AclDataType<int64_t>::type)).Run();
        index_int64.print();
        NpuTensor<const int32_t> axis({1}, {1});

        NpuRunner runner("GatherElements");
        runner.AddInput(in).AddInput(index).AddOutput(out).SetAttr("dim", 2).Run();
      };

      NpuTensor<int32_t> output_0_2({2, 2, 2});
      gether_elements_op(ids_0_2, parent_1_2, parent_1_2.dims, output_0_2);
      output_0_2.print();
    }
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
