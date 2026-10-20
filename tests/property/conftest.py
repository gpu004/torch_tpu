"""Import real TorchTPU Python modules without loading the TPU extension."""

from __future__ import annotations

import enum
import pathlib
import sys
import types


_SOURCE = pathlib.Path(__file__).resolve().parents[2] / "src" / "torch_tpu"


def _namespace(name: str, path: pathlib.Path) -> types.ModuleType:
  module = types.ModuleType(name)
  module.__path__ = [str(path)]
  module.__package__ = name
  sys.modules[name] = module
  return module


_torch_tpu = _namespace("torch_tpu", _SOURCE)
_internal = _namespace("torch_tpu._internal", _SOURCE / "_internal")
_torch_tpu._internal = _internal

_compile = _namespace(
    "torch_tpu._internal.compile", _SOURCE / "_internal" / "compile"
)
_internal.compile = _compile
_compile.dynamic = _namespace(
    "torch_tpu._internal.compile.dynamic",
    _SOURCE / "_internal" / "compile" / "dynamic",
)
for _name in ("distributed", "ops", "utils"):
  _module = _namespace(
      f"torch_tpu._internal.{_name}", _SOURCE / "_internal" / _name
  )
  setattr(_internal, _name, _module)


class _WarningCategory(enum.Enum):
  TORCH_TPU = "TorchTpu"
  PERFORMANCE = "Performance"
  UNOPTIMIZED_BUILD = "UnoptimizedBuild"
  TCMALLOC = "TcMalloc"
  CORRECTNESS = "Correctness"
  API_STAGE = "ApiStage"
  INTERNAL_API = "InternalApi"
  EXPERIMENTAL_API = "ExperimentalApi"
  DEPRECATED_API = "DeprecatedApi"


_warnings_impl = types.ModuleType("torch_tpu._internal.warnings_impl")
_warnings_impl.WarningCategory = _WarningCategory
_warnings_impl.get_warning_prefix = lambda category: (
    f"<TorchTpu/{category.value}/>"
)
_warnings_impl.is_deprecation_warning = lambda category: (
    category == _WarningCategory.DEPRECATED_API
)
sys.modules[_warnings_impl.__name__] = _warnings_impl
_internal.warnings_impl = _warnings_impl

_tpu_distributed = types.ModuleType(
    "torch_tpu._internal.distributed.tpu_distributed"
)
_tpu_distributed.global_device_attributes = lambda: []
_tpu_distributed.rank_to_device_id = lambda backend: []
sys.modules[_tpu_distributed.__name__] = _tpu_distributed
sys.modules["torch_tpu._internal.distributed"].tpu_distributed = (
    _tpu_distributed
)


EXPRESSIBLE_VIEW_NOT_IMPLEMENTED: list[tuple[tuple[int, ...], tuple[int, ...], int]] = []


def pytest_terminal_summary(terminalreporter, exitstatus, config):
  del exitstatus, config
  if EXPRESSIBLE_VIEW_NOT_IMPLEMENTED:
    examples = sorted(
        set(EXPRESSIBLE_VIEW_NOT_IMPLEMENTED),
        key=lambda case: (case[0], case[1], case[2]),
    )[:5]
    terminalreporter.write_sep(
        "-", "slice(step=1)+permute views rejected with NotImplementedError"
    )
    for shape, strides, offset in examples:
      terminalreporter.write_line(
          f"shape={shape}, strides={strides}, storage_offset={offset}"
      )
