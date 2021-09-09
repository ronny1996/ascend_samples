cp /usr/local/Ascend/ascend-toolkit/latest/toolkit/python/site-packages/op_gen/json_template/IR_json.json ./
# msopgen gen -i json_path/IR_json.json -f tf -c ai_core-{Soc Version} -out ./output_data
# ll /usr/local/Ascend/ascend-toolkit/5.0.2.alpha005/x86_64-linux/fwkacllib/data/platform_config/*.ini
msopgen gen -i ./IR_json.json -f tf -c ai_core-Ascend910A -out ./output_data
ll output_data/op_proto/ #算子原型，实现*.cc中的方法
ll tbe/impl/*.py #实现算子
ll build.sh #修改build.sh
./build.sh

# OPP组件的安装路径
export ASCEND_OPP_PATH=/home/HwHiAiUser/Ascend/ascend-toolkit/latest/opp
# AI CPU
export ASCEND_AICPU_PATH=/home/HwHiAiUser/Ascend/ascend-toolkit/latest
export ASCEND_TENSOR_COMPILER_INCLUDE=/home/HwHiAiUser/Ascend/ascend-toolkit/latest/atc/include
export TOOLCHAIN_DIR=/home/HwHiAiUser/Ascend/ascend-toolkit/latest/toolkit/toolchain/hcc
export AICPU_KERNEL_TARGET=cust_aicpu_kernels_3.3.0
export AICPU_SOC_VERSION=Ascend310


# install
/usr/local/Ascend/nnae/latest/opp/framework
/usr/local/Ascend/nnae/latest/opp/op_proto
/usr/local/Ascend/nnae/latest/opp/op_impl

# /usr/local/Ascend/ascend-toolkit/latest/opp/framework/custom/
