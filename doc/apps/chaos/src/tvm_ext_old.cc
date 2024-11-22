#include <iostream>
#include <vector>
#include <string.h>
#include <sstream>
#include <cassert> // 断言
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/logging.h>

using namespace tvm::runtime;

void fuzz(TVMArgs args, TVMRetValue* rv) {
  // 自动将参数转换为所需的类型。
  int a = args[0];
  int b = args[1];
  LOG(INFO) << "args: " << a << " " << b;
  // 自动分配返回值 rv
  *rv = a * b;
}

// 注册全局 packed function
TVM_REGISTER_GLOBAL("fuzz").set_body(fuzz);


std::pair<int32_t, int32_t> GetFixedPointMultiplierShift_Uint8(double double_multiplier) {
  int32_t significand, exponent;
  if (double_multiplier == 0.) {
    significand = 0;
    exponent = 0;
    return std::make_pair(significand, exponent);
  }
  //浮点数:
  //https://blog.csdn.net/weixin_45863060/article/details/125054244
  //把浮点数分解成尾数和阶码,其中尾数是小于1的小数
  //把尾数调整乘以256,保证转出来的整数在uint8范围内
  //保证乘法在int32下不溢出
  double significand_d = std::frexp(double_multiplier, &exponent);
  // significand = std::round((significand_d + 1)*(1ll << 7));	//因为uint8
  significand = std::round(significand_d*255+0.5);	//因为uint8
  assert(significand <= 255);
  //???照猫画虎???
  if (significand == 255) { 						
    significand /= 2;		
    ++exponent;
  }
  return std::make_pair(significand, exponent);
}

std::pair<int32_t, int32_t> GetFixedPointMultiplierShift_Int8(double double_multiplier)
{
  int32_t significand, exponent;
  if (double_multiplier == 0.) {
    significand = 0;
    exponent = 0;
    return std::make_pair(significand, exponent);
  }
  //浮点数:
  //https://blog.csdn.net/weixin_45863060/article/details/125054244
  //把浮点数分解成尾数和阶码,其中尾数是小于1的小数
  //把尾数调整乘以256,保证转出来的整数在uint8范围内
  //保证乘法在int32下不溢出
  double significand_d = std::frexp(double_multiplier, &exponent);
  significand = std::round(significand_d*128+0.5);		//因为int8
  //???照猫画虎???
  if (significand == 128) { 						
    significand /= 2;		
    ++exponent;
  }
  return std::make_pair(significand, exponent);
}
