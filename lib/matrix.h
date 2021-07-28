#ifndef FAST_MATMUL_LIB_MATRIX_H
#define FAST_MATMUL_LIB_MATRIX_H

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>

#include "arithmetic.h"

namespace fastmm {

template <Arithmetic T>
class Matrix {
public:
  using row_type     = typename std::vector<T>;
  using row_iterator = typename std::vector<row_type>::iterator;
  using value_type   = T;

  Matrix();

  explicit Matrix(size_t n);

  Matrix(size_t m, size_t n);

  Matrix(size_t m, size_t n, T value);

  Matrix(std::initializer_list<std::initializer_list<T>> init);

  [[nodiscard]] auto begin() const;

  [[nodiscard]] auto end() const;

  auto fill(T value) -> void;

  [[nodiscard]] auto height() const -> size_t;

  [[nodiscard]] auto shape() const -> std::pair<size_t, size_t>;

  [[nodiscard]] auto transpose() const -> Matrix;

  [[nodiscard]] auto width() const -> size_t;

  [[nodiscard]] auto operator[](size_t) -> std::vector<T>&;

  [[nodiscard]] auto operator[](size_t) const -> const std::vector<T>&;

  [[nodiscard]] auto operator*(Matrix&) -> Matrix;

private:
  std::vector<std::vector<T>> m_data;

}; // class Matrix

template <Arithmetic T>
Matrix<T>::Matrix() = default;

template <Arithmetic T>
Matrix<T>::Matrix(size_t n)
  : m_data(n, std::vector<T>(n))
{}

template <Arithmetic T>
Matrix<T>::Matrix(size_t m, size_t n)
  : m_data(m, std::vector<T>(n))
{}

template <Arithmetic T>
Matrix<T>::Matrix(size_t m, size_t n, T value)
  : Matrix(m, n)
{
  fill(value);
}

template <Arithmetic T>
Matrix<T>::Matrix(std::initializer_list<std::initializer_list<T>> init)
  : Matrix { init.size(), std::begin(init)->size() }
{
  size_t i = 0;
  size_t j = 0;
  for (const auto& row : init) {
    for (const auto el : row) {
      m_data[i][j] = el;
      ++j;
    }
    j = 0;
    ++i;
  }
}

template <Arithmetic T>
[[nodiscard]]
auto Matrix<T>::begin() const {
  return m_data.begin();
}

template <Arithmetic T>
[[nodiscard]]
auto Matrix<T>::end() const {
  return m_data.end();
}

template <Arithmetic T>
auto Matrix<T>::fill(T value) -> void {
  for (auto& row : m_data) {
    std::fill(std::begin(row), std::end(row), value);
  }
}

template <Arithmetic T>
[[nodiscard]]
auto Matrix<T>::height() const -> size_t {
  return m_data.size();
}

template <Arithmetic T>
[[nodiscard]]
auto Matrix<T>::shape() const -> std::pair<size_t, size_t> {
  return { height(), width() };
}

template <Arithmetic T>
[[nodiscard]]
auto Matrix<T>::transpose() const -> Matrix<T> {
  const auto [m, n] = shape();
  Matrix transposed(n, m);
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      transposed[j][i] = m_data[i][j];
    }
  }
  return transposed;
}

template <Arithmetic T>
[[nodiscard]]
auto Matrix<T>::width() const -> size_t {
  return m_data.front().size();
}

template <Arithmetic T>
[[nodiscard]]
auto Matrix<T>::operator[](size_t i) -> std::vector<T>& {
  return m_data[i];
}

template <Arithmetic T>
[[nodiscard]]
auto Matrix<T>::operator[](size_t i) const -> const std::vector<T>& {
  return m_data[i];
}

template <Arithmetic T>
[[nodiscard]]
auto Matrix<T>::operator*(Matrix<T>& rhs) -> Matrix<T> {
  const auto [m1, n1] = shape();
  const auto [m2, n2] = rhs.shape();
  auto&& rhs_t = rhs.transpose();
  Matrix<T> product(m1, n2);
  for (size_t i = 0; i < m1; ++i) {
    const auto& row = (*this)[i];
    for (size_t j = 0; j < n2; ++j) {
      const auto& col = rhs_t[j];
      product[i][j] = std::inner_product(std::begin(row), std::end(row), std::begin(col), T {});
    }
  }
  return product;
}

} // namespace fastmm

#endif // FAST_MATMUL_LIB_MATRIX_H
