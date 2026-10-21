#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <absl/log/log_sink.h>
#include <absl/log/log_sink_registry.h>
#include <absl/strings/str_cat.h>
#include <hegel/hegel.h>

#if defined(HEGEL_VERSION_TARGET)
#include "csrc/common/version.h"
#elif defined(HEGEL_CONSTEXPR_MAP_TARGET)
#include "csrc/common/constexpr_map.h"
#elif defined(HEGEL_UTILS_TARGET)
#include "csrc/common/utils.h"
#endif

namespace gs = hegel::generators;
#if defined(HEGEL_VERSION_TARGET)
using torch_tpu::Version;
#endif

namespace {

#if defined(HEGEL_VERSION_TARGET)
bool IsDigit(char c) { return c >= '0' && c <= '9'; }

template <typename T>
bool ParseDigits(std::string_view text, T* value) {
  if (text.empty()) return false;
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), *value);
  return error == std::errc{} && end == text.data() + text.size();
}

std::optional<Version> ReferenceParse(std::string_view input) {
  size_t pos = 0;
  auto read_component = [&](auto* value) {
    const size_t begin = pos;
    while (pos < input.size() && IsDigit(input[pos])) ++pos;
    return ParseDigits(input.substr(begin, pos - begin), value);
  };

  int major = 0;
  int minor = 0;
  int patch = 0;
  int64_t dev = Version::kStable;
  if (!read_component(&major)) return std::nullopt;

  if (pos < input.size() && input[pos] == '.') {
    if (input.substr(pos).starts_with(".dev")) {
      pos += 4;
      if (!read_component(&dev)) return std::nullopt;
    } else {
      ++pos;
      if (!read_component(&minor)) return std::nullopt;
      if (pos < input.size() && input[pos] == '.') {
        if (input.substr(pos).starts_with(".dev")) {
          pos += 4;
          if (!read_component(&dev)) return std::nullopt;
        } else {
          ++pos;
          if (!read_component(&patch)) return std::nullopt;
        }
      }
      if (pos < input.size() && input.substr(pos).starts_with(".dev")) {
        pos += 4;
        if (!read_component(&dev)) return std::nullopt;
      }
    }
  }

  if (pos < input.size()) {
    if (IsDigit(input[pos]) || input[pos] == '.') return std::nullopt;
    pos = input.size();
  }
  return Version(major, minor, patch, dev);
}
#endif

#if defined(HEGEL_UTILS_TARGET)
std::string ToString128(unsigned __int128 value) {
  if (value == 0) return "0";
  std::string out;
  while (value != 0) {
    out.push_back(static_cast<char>('0' + value % 10));
    value /= 10;
  }
  std::reverse(out.begin(), out.end());
  return out;
}
#endif

#if defined(HEGEL_CONSTEXPR_MAP_TARGET)
struct AbsoluteLess {
  constexpr bool operator()(int lhs, int rhs) const {
    return std::abs(lhs) < std::abs(rhs);
  }
};

template <typename Map, typename Oracle>
void CompareMaps(const Map& actual, const Oracle& expected,
                 const std::vector<int>& queries) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error("map size differs from std::map");
  }
  auto actual_it = actual.begin();
  auto expected_it = expected.begin();
  for (; expected_it != expected.end(); ++expected_it, ++actual_it) {
    if (actual_it->key != expected_it->first ||
        actual_it->value != expected_it->second) {
      throw std::runtime_error("map iteration differs from std::map");
    }
  }
  for (int key : queries) {
    const auto found = actual.find(key);
    const auto oracle = expected.find(key);
    if ((found == actual.end()) != (oracle == expected.end())) {
      throw std::runtime_error("map find differs from std::map");
    }
    if (found != actual.end()) {
      if (found->value != oracle->second || actual.at(key) != oracle->second ||
          !actual.contains(key)) {
        throw std::runtime_error("map access differs from std::map");
      }
    } else if (actual.contains(key)) {
      throw std::runtime_error("map contains differs from std::map");
    }
  }
}
#endif

#if defined(HEGEL_UTILS_TARGET)
class CapturingLogSink : public absl::LogSink {
 public:
  void Send(const absl::LogEntry& entry) override {
    messages.emplace_back(entry.text_message());
  }

  std::vector<std::string> messages;
};
#endif

}

#if defined(HEGEL_VERSION_TARGET)
HEGEL_TEST(version_stringify_round_trip)(
    hegel::TestCase& tc) {
  const int major = tc.draw(
      "major", gs::integers<int>({.min_value = 0,
                                 .max_value = std::numeric_limits<int>::max()}));
  const int minor = tc.draw(
      "minor", gs::integers<int>({.min_value = 0,
                                 .max_value = std::numeric_limits<int>::max()}));
  const int patch = tc.draw(
      "patch", gs::integers<int>({.min_value = 0,
                                 .max_value = std::numeric_limits<int>::max()}));
  const int64_t build = tc.draw(
      "build", gs::integers<int64_t>({.min_value = 0,
                                     .max_value = 999999999999LL}));
  const bool is_dev = tc.draw(gs::booleans());
  const Version expected(major, minor, patch,
                         is_dev ? build : Version::kStable);
  const auto parsed = Version::Parse(absl::StrCat(expected));
  if (!parsed.ok() || *parsed != expected) {
    throw std::runtime_error("Parse(AbslStringify(version)) changed version");
  }
}

HEGEL_TEST(version_parser_matches_reference)(
    hegel::TestCase& tc) {
  const int major = tc.draw(
      "major", gs::integers<int>({.min_value = 0, .max_value = 999999}));
  const int minor = tc.draw(
      "minor", gs::integers<int>({.min_value = 0, .max_value = 999999}));
  const int patch = tc.draw(
      "patch", gs::integers<int>({.min_value = 0, .max_value = 999999}));
  const int64_t build = tc.draw(
      "build", gs::integers<int64_t>({.min_value = 0,
                                     .max_value = 999999999999LL}));
  const int shape = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 5}));
  const std::vector<std::string> suffixes = {
      "", "+nightly", "rc1", "1", ".", ".bad", "x.2", ".dev", ".dev1"};
  const auto suffix =
      suffixes[tc.draw(gs::integers<size_t>(
          {.min_value = 0, .max_value = suffixes.size() - 1}))];
  std::string input = std::to_string(major);
  if (shape >= 1) input += "." + std::to_string(minor);
  if (shape >= 2) input += "." + std::to_string(patch);
  if (shape == 3) input += ".dev" + std::to_string(build);
  input += suffix;

  const auto expected = ReferenceParse(input);
  const auto actual = Version::Parse(input);
  if (actual.ok() != expected.has_value()) {
    throw std::runtime_error("Parse disagrees with grammar for: " + input);
  }
  if (expected && *actual != *expected) {
    throw std::runtime_error("Parse disagrees with reference value for: " +
                             input);
  }
}

HEGEL_TEST(version_ordering_matches_tuple_order)(
    hegel::TestCase& tc) {
  const auto draw_version = [&]() {
    const int major = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 20}));
    const int minor = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 20}));
    const int patch = tc.draw(gs::integers<int>({.min_value = 0, .max_value = 20}));
    const int64_t dev = tc.draw(
        gs::integers<int64_t>({.min_value = 0, .max_value = 999999999999LL}));
    return std::pair{Version(major, minor, patch, dev),
                     std::tuple{major, minor, patch, dev}};
  };
  const auto [lhs, lhs_tuple] = draw_version();
  const auto [rhs, rhs_tuple] = draw_version();
  if ((lhs <=> rhs) != (lhs_tuple <=> rhs_tuple)) {
    throw std::runtime_error("Version ordering differs from tuple ordering");
  }
}
#elif defined(HEGEL_CONSTEXPR_MAP_TARGET)

HEGEL_TEST(constexpr_map_matches_std_map)(
    hegel::TestCase& tc) {
  constexpr size_t kSize = 8;
  using Map = torch_tpu::ConstexprMap<int, int, kSize>;
  Map::value_type entries[kSize];
  std::map<int, int> expected;
  for (size_t i = 0; i < kSize; ++i) {
    const int key = tc.draw(gs::integers<int>({.min_value = -20,
                                              .max_value = 20}));
    const int value = tc.draw(gs::integers<int>({.min_value = -100,
                                                .max_value = 100}));
    entries[i] = {key, value};
    expected.emplace(key, value);
  }
  const std::vector<int> queries = {
      tc.draw(gs::integers<int>({.min_value = -25, .max_value = 25})),
      -21,
      0,
      21,
  };
  if (expected.size() != kSize) {
    try {
      const Map duplicate_check(entries);
      (void)duplicate_check;
    } catch (const std::invalid_argument&) {
      return;
    }
    throw std::runtime_error("ConstexprMap accepted duplicate keys");
  }
  const Map actual(entries);
  CompareMaps(actual, expected, queries);
}

HEGEL_TEST(constexpr_map_custom_comparator)(
    hegel::TestCase& tc) {
  constexpr size_t kSize = 8;
  using Map = torch_tpu::ConstexprMap<int, int, kSize, AbsoluteLess>;
  Map::value_type entries[kSize];
  std::map<int, int, AbsoluteLess> expected;
  for (size_t i = 0; i < kSize; ++i) {
    const int key = tc.draw(gs::integers<int>({.min_value = -20,
                                              .max_value = 20}));
    const int value = tc.draw(gs::integers<int>({.min_value = -100,
                                                .max_value = 100}));
    entries[i] = {key, value};
    expected.emplace(key, value);
  }
  const std::vector<int> queries = {-10, -1, 0, 1, 10};
  if (expected.size() != kSize) {
    try {
      const Map duplicate_check(entries);
      (void)duplicate_check;
    } catch (const std::invalid_argument&) {
      return;
    }
    throw std::runtime_error("ConstexprMap accepted comparator-equivalent keys");
  }
  const Map actual(entries);
  CompareMaps(actual, expected, queries);
}
#elif defined(HEGEL_UTILS_TARGET)

HEGEL_TEST(percentage_matches_exact_integer_arithmetic)(
    hegel::TestCase& tc) {
  const uint64_t numerator = tc.draw(gs::integers<uint64_t>(
      {.min_value = 0,
       .max_value = std::numeric_limits<uint64_t>::max() / 1000}));
  const uint64_t denominator = tc.draw(gs::integers<uint64_t>());
  if (denominator == 0) {
    if (torch_tpu::PercAsStr(numerator, denominator) != "NA") {
      throw std::runtime_error("zero denominator did not produce NA");
    }
    return;
  }
  const unsigned __int128 scaled =
      static_cast<unsigned __int128>(numerator) * 1000 / denominator;
  const std::string expected =
      ToString128(scaled / 10) + "." +
      std::to_string(static_cast<unsigned>(scaled % 10)) + "%";
  const auto actual = torch_tpu::PercAsStr(numerator, denominator);
  if (actual != expected) {
    tc.note("numerator=" + std::to_string(numerator) +
            " denominator=" + std::to_string(denominator) +
            " expected=" + expected + " actual=" + actual);
    throw std::runtime_error("PercAsStr differs from exact arithmetic");
  }
}

HEGEL_TEST(log_lines_preserves_line_content_and_order)(
    hegel::TestCase& tc) {
  const auto lines = tc.draw(
      gs::vectors(gs::integers<int>({.min_value = 0, .max_value = 999999}),
                  {.min_size = 1, .max_size = 100}));
  std::string input;
  for (int line : lines) input += "line-" + std::to_string(line) + "\n";

  CapturingLogSink sink;
  absl::AddLogSink(&sink);
  torch_tpu::LogLines(input);
  absl::RemoveLogSink(&sink);

  std::string captured;
  for (const auto& message : sink.messages) captured += message;
  size_t position = 0;
  for (int line : lines) {
    const auto token = "line-" + std::to_string(line);
    const auto found = captured.find(token, position);
    if (found == std::string::npos) {
      throw std::runtime_error("LogLines dropped or reordered content");
    }
    position = found + token.size();
  }
}
#endif

int main(int argc, char** argv) {
  const std::string_view selector = argc > 1 ? argv[1] : "";
  const bool has_property_selector =
      selector == "percentage" || selector == "log";
  const int cases_arg = has_property_selector ? 2 : 1;
  const auto run = [argc, argv, cases_arg](auto test) {
    if (argc > cases_arg) {
      test({.test_cases = std::stoull(argv[cases_arg])});
    } else {
      test({});
    }
  };
#if defined(HEGEL_VERSION_TARGET)
  run(version_stringify_round_trip);
  run(version_parser_matches_reference);
  run(version_ordering_matches_tuple_order);
#elif defined(HEGEL_CONSTEXPR_MAP_TARGET)
  run(constexpr_map_matches_std_map);
  run(constexpr_map_custom_comparator);
#elif defined(HEGEL_UTILS_TARGET)
  if (selector == "percentage") {
    run(percentage_matches_exact_integer_arithmetic);
  } else if (selector == "log") {
    run(log_lines_preserves_line_content_and_order);
  } else {
    run(percentage_matches_exact_integer_arithmetic);
    run(log_lines_preserves_line_content_and_order);
  }
#endif
  return 0;
}
