#ifndef FAST_MATMUL_LIB_ARITHMETIC_H
#define FAST_MATMUL_LIB_ARITHMETIC_H

#include <type_traits>

namespace fastmm {

template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

} // namespace fastmm


#endif // FAST_MATMUL_LIB_ARITHMETIC_H
