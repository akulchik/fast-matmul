#ifndef FAST_MATMUL_LIB_MATMUL_H
#define FAST_MATMUL_LIB_MATMUL_H

#include "matrix.h"

namespace fastmm {

template <Arithmetic T>
[[nodiscard]]
auto matmul_v0(const Matrix<T>& a, const Matrix<T>& b) -> Matrix<T>;

template <Arithmetic T>
[[nodiscard]]
auto matmul_v0(const Matrix<T>& a, const Matrix<T>& b) -> Matrix<T> {
  const auto [m, p] = a.shape();
  const auto [s, n] = b.width();
  assert(p == s);
  auto c = Matrix<T>(m, n, 0);
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      c[i][j] = 0;
      for (size_t k = 0; k < p; ++k) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  return c;
}

} // namespace fastmm

#endif // FAST_MATMUL_LIB_MATMUL_H
