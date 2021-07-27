
.PHONY : all

CXXFLAGS:=
CXXFLAGS+=-fpermissive -Wno-deprecated-declarations

LDFLAGS:=
LDFLAGS+=-L/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/fwkacllib/lib64/

INCFLAGS:=
INCFLAGS+=-I/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/fwkacllib/include/

LIBS:=
LIBS+=-lascendcl -lacl_op_compiler

CXX_OBJECT:= $(patsubst %.cc, %.o, $(wildcard *.cc))
CXX_TARGET:= $(CXX_OBJECT:.o=)

all : ${CXX_TARGET}
	@echo "Build targets: ${CXX_TARGET}"

${CXX_OBJECT}: %.o : %.cc
	$(CXX) -c ${CXXFLAGS} ${INCFLAGS} $< -o $@ 

${CXX_TARGET}: % : %.o
	$(CXX) ${CXXFLAGS} ${INCFLAGS} ${LDFLAGS} ${LIBS} $< -o $@ 

clean :
	@rm -rf ${CXX_TARGET}
	@rm -rf ${CXX_OBJECT}
	@rm -rf kernel_meta fusion_result.json ge_check_op.json
	@rm -rf test/kernel_meta/ test/fusion_result.json test/ge_check_op.json
