#include "npu_runner.h"

int main(int argc, char const* argv[]) {
  npu::float16 a(1.23f);
  std::cout << sizeof(a) << std::endl;
  std::cout << static_cast<float>(a) << std::endl;
}
