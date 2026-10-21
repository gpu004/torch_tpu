#pragma once

#include <sstream>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace torch_tpu::error {
inline constexpr absl::StatusCode kInvalidArgument =
    absl::StatusCode::kInvalidArgument;
inline constexpr absl::StatusCode kPythonNotImplementedError =
    absl::StatusCode::kUnimplemented;
}

namespace torch_tpu::property_test {

class StatusBuilder {
 public:
  explicit StatusBuilder(absl::StatusCode code) : code_(code) {}
  explicit StatusBuilder(absl::Status status)
      : code_(status.code()), prefix_(status.message()) {}

  template <typename T>
  StatusBuilder& operator<<(T&& value) {
    stream_ << std::forward<T>(value);
    return *this;
  }

  operator absl::Status() { return {code_, stream_.str()}; }

  template <typename T>
  operator absl::StatusOr<T>() {
    return absl::Status(code_, prefix_ + stream_.str());
  }

 private:
  absl::StatusCode code_;
  std::string prefix_;
  std::ostringstream stream_;
};

}

#define TT_RET_CHECK(condition, code)               \
  if (condition) {                                   \
  } else                                             \
    return ::torch_tpu::property_test::StatusBuilder(code)

#define TT_ERROR(code) ::torch_tpu::property_test::StatusBuilder(code)

#define TT_RETURN_IF_ERROR(expression) \
  ::torch_tpu::property_test::StatusBuilder(expression)

#define TT_STATUS_CONCAT_INNER(lhs, rhs) lhs##rhs
#define TT_STATUS_CONCAT(lhs, rhs) TT_STATUS_CONCAT_INNER(lhs, rhs)
#define TT_ASSIGN_OR_RETURN_IMPL(lhs, expression, id)     \
  auto TT_STATUS_CONCAT(_tt_status_or_, id) = (expression); \
  if (!TT_STATUS_CONCAT(_tt_status_or_, id).ok()) {       \
    return TT_STATUS_CONCAT(_tt_status_or_, id).status(); \
  }                                                        \
  lhs = std::move(TT_STATUS_CONCAT(_tt_status_or_, id)).value()
#define TT_ASSIGN_OR_RETURN(lhs, expression) \
  TT_ASSIGN_OR_RETURN_IMPL(lhs, expression, __COUNTER__)
