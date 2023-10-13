#include <iostream>
#include <vector>
#include <string.h>
#include <sstream>
#include <cassert> // 断言

typedef int64_t index_t;   // 用于标识张量索引的类型
typedef uint32_t shape_t;  // 用于标识张量形状的类型

int main(void) {
  std::ostringstream oss;  // 用于记录信息
  oss << "ddd" << "\n";
  std::cout << oss.str();
  oss.str(""); // 用于清除缓存
  return 0;
}
