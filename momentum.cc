#include "npu_runner.h"

// vt = momentum * vt_1 - learning_rate * grad
// var = var + vt


int main(int argc, char const* argv[]) {
  /* code */
  NpuHelper::InitAllDevices();
  NpuHelper::SetDevice(0);
  #if 0
  {
    NpuTensor<float> var_tensor({5}, {1, 1, 1, 1, 1});
    NpuTensor<float> accum_tensor({5}, {0, 0, 0, 0, 0});
    NpuTensor<float> lr_tensor({1}, {0.8});
    NpuTensor<float> grad_tensor({5}, {2, 2, 2, 2, 2});
    NpuTensor<float> momentum_tensor({1}, {0.9});
    {
      /**
       * Updates '*var' according to the momentum scheme. 
       * accum = accum * momentum - grad * lr 
       * if use_nesterov is True: var += accum * momentum - grad * lr else: var += accum.
       **/

      NpuRunner runner("ApplyKerasMomentum");
      runner.AddInput(var_tensor)
          .AddInput(accum_tensor)
          .AddInput(lr_tensor)
          .AddInput(grad_tensor)
          .AddInput(momentum_tensor)
          .AddOutput(var_tensor)
          .SetAttr("use_nesterov", false)
          .Run();
    }
    var_tensor.print();
    accum_tensor.print();
  }
  #endif
  {
    NpuTensor<float> var_tensor({5}, std::vector<float>(5, 1.0f));
    NpuTensor<float> accum_tensor({5}, std::vector<float>(5, 0.0f));
    NpuTensor<float> lr_tensor({1}, {0.001});
    NpuTensor<float> grad_tensor({5}, std::vector<float>(5, 2.0f));
    NpuTensor<float> momentum_tensor({1}, {0.0001});
    NpuTensor<float> var_out_tensor({5});
    {
      NpuRunner runner("ApplyMomentum");
      runner.AddInput(var_tensor)
          .AddInput(accum_tensor)
          .AddInput(lr_tensor)
          .AddInput(grad_tensor)
          .AddInput(momentum_tensor)
          .AddOutput(var_out_tensor)
          .SetAttr("use_nesterov", false)
          .SetAttr("use_locking", true)
          .Run();
    }
    var_tensor.print();
    var_out_tensor.print();
    accum_tensor.print();
  }
  NpuHelper::ReleaseAllDevices();
  return 0;
}
