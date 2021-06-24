import paddle
import numpy as np
from paddle.fluid.layer_helper import LayerHelper

paddle.enable_static()

exe = paddle.static.Executor(paddle.NPUPlace(0))
main_program = paddle.static.Program()
startup_program = paddle.static.Program()

def gaussian():
    helper = LayerHelper('gaussian_random', **locals())
    attrs = {'shape': [5], 'mean': 2.0, 'std': 1.0, 'seed': 0}
    out = helper.create_variable_for_type_inference(np.float32)
    helper.append_op(
        type='gaussian_random', inputs={}, outputs={'Out': out}, attrs=attrs)
    return out


with paddle.static.program_guard(main_program=main_program, startup_program=startup_program):
    out = gaussian()

startup_program.random_seed = 1
exe.run(startup_program)
outs = exe.run(main_program, fetch_list=[out.name])
print(outs)
