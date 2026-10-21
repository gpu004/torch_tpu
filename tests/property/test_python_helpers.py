"""Property tests for pure Python TorchTPU helpers."""

from __future__ import annotations

import math
import random
import types
from unittest import mock

from hypothesis import assume, given, settings, strategies as st
import pytest
import sympy
import torch
from torch.utils._sympy import value_ranges

from torch_tpu._internal.compile.dynamic import symbol_bounds
from torch_tpu._internal.distributed import device_mesh
from torch_tpu._internal.ops import jagged
from torch_tpu._internal.utils import hardware


_DTYPES = (torch.bool, torch.int8, torch.int32, torch.float32, torch.bfloat16)


@st.composite
def _jagged_cases(draw):
  count = draw(st.integers(min_value=1, max_value=6))
  lengths = draw(
    st.lists(
      st.integers(min_value=0, max_value=50),
      min_size=count,
      max_size=count,
    )
  )
  inner_shape = draw(
    st.one_of(
      st.just(()),
      st.tuples(
        st.integers(min_value=1, max_value=4),
        st.integers(min_value=1, max_value=3),
      ),
    )
  )
  dtype = draw(st.sampled_from(_DTYPES))
  pieces = []
  for length in lengths:
    shape = (length,) + inner_shape
    if dtype == torch.bool:
      values = draw(
        st.lists(
          st.booleans(),
          min_size=math.prod(shape),
          max_size=math.prod(shape),
        )
      )
    elif dtype in (torch.int8, torch.int32):
      values = draw(
        st.lists(
          st.integers(min_value=-100, max_value=100),
          min_size=math.prod(shape),
          max_size=math.prod(shape),
        )
      )
    else:
      values = draw(
        st.lists(
          st.floats(
            min_value=-10,
            max_value=10,
            allow_nan=False,
            allow_infinity=False,
          ),
          min_size=math.prod(shape),
          max_size=math.prod(shape),
        )
      )
    pieces.append(torch.tensor(values, dtype=dtype).reshape(shape))
  multiple = draw(st.integers(min_value=1, max_value=200))
  return torch.nested.nested_tensor(pieces, layout=torch.jagged), multiple


@settings(max_examples=150, deadline=None)
@given(case=_jagged_cases())
def test_jagged_align_and_strip_round_trip(case):
  nested, multiple = case
  original_values = nested.values()
  original_offsets = nested.offsets()
  valid_tokens = original_values.size(0)

  padded = jagged.align_jagged_to_multiple(nested, multiple)
  assert padded.values().size(0) % multiple == 0
  assert torch.equal(padded.offsets(), original_offsets)

  for valid_count in (None, valid_tokens):
    stripped = jagged.strip_jagged_padding(padded, valid_count)
    assert torch.equal(stripped.values(), original_values)
    assert torch.equal(stripped.offsets(), original_offsets)

  unpadded = jagged.strip_jagged_padding(nested)
  assert unpadded is nested


@pytest.mark.xfail(strict=True, reason="google-pytorch/torch_tpu#5673")
def test_align_jagged_to_multiple_preserves_explicit_lengths():
  nested = torch.nested.nested_tensor_from_jagged(
    torch.arange(10.0).reshape(10, 1),
    torch.tensor([0, 4, 10]),
    lengths=torch.tensor([2, 3]),
  )
  original_sequences = nested.unbind()

  aligned = jagged.align_jagged_to_multiple(nested, 4)

  assert torch.equal(aligned.lengths(), nested.lengths())
  aligned_sequences = aligned.unbind()
  assert len(aligned_sequences) == len(original_sequences)
  assert all(
    torch.equal(actual, expected)
    for actual, expected in zip(aligned_sequences, original_sequences)
  )


@settings(max_examples=150, deadline=None)
@given(
  extents=st.lists(
    st.integers(min_value=1, max_value=16), min_size=1, max_size=4
  ),
  slices=st.integers(min_value=1, max_value=16),
)
def test_inferred_dcn_shape_is_accepted(extents, slices):
  axis = next((extent for extent in extents if extent > 1), None)
  assume(axis is not None and axis % slices == 0)
  inferred = device_mesh._infer_dcn_mesh_shape(tuple(extents), slices)
  per_slice = math.prod(extents) // slices
  infos = [
    device_mesh._DeviceStub(
      id=rank,
      coords=(rank, 0, 0),
      core_on_chip=0,
      device_kind="TPU v5e",
      platform="tpu",
      process_index=rank,
      slice_index=rank // per_slice,
    )
    for rank in range(math.prod(extents))
  ]
  device_mesh._validate_multi_slice(infos, tuple(extents), inferred)


def _v5e_devices(num_slices, per_slice):
  coords = (
    [(x, y, 0) for x in range(2) for y in range(2)]
    if per_slice == 4
    else [(x, y, 0) for x in range(2) for y in range(4)]
  )
  devices = []
  for slice_index in range(num_slices):
    for local_id, coordinate in enumerate(coords):
      device_id = slice_index * per_slice + local_id
      devices.append(
        device_mesh._DeviceStub(
          id=device_id,
          coords=coordinate,
          core_on_chip=0,
          device_kind="TPU v5e",
          platform="tpu",
          process_index=device_id,
          slice_index=slice_index,
        )
      )
  return devices


@settings(max_examples=50, deadline=None)
@given(
  num_slices=st.integers(min_value=2, max_value=4),
  per_slice=st.sampled_from((4, 8)),
  seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_topology_aware_mesh_keeps_slices_on_dcn_axis(
  num_slices, per_slice, seed
):
  devices = _v5e_devices(num_slices, per_slice)
  random.Random(seed).shuffle(devices)
  mesh_shape = (num_slices, per_slice)
  with (
    mock.patch.object(device_mesh.dist, "is_initialized", return_value=True),
    mock.patch.object(
      device_mesh.dist, "get_world_size", return_value=len(devices)
    ),
    mock.patch.object(
      device_mesh.tpu_distributed,
      "global_device_attributes",
      return_value=devices,
    ),
  ):
    rank_mesh = device_mesh.topology_aware_mesh("tpu", mesh_shape)
  assert sorted(rank_mesh.flatten().tolist()) == list(range(len(devices)))

  slice_by_rank = {
    rank: device.slice_index for rank, device in enumerate(devices)
  }
  for dcn_coordinate in range(num_slices):
    slice_ids = {
      slice_by_rank[rank]
      for rank in rank_mesh[dcn_coordinate].flatten().tolist()
    }
    assert len(slice_ids) == 1


@settings(max_examples=100, deadline=None)
@given(
  slice_count=st.integers(min_value=1, max_value=8),
  local_count=st.integers(min_value=1, max_value=64),
)
def test_validate_multi_slice_accepts_consistent_shape(
  slice_count, local_count
):
  total = slice_count * local_count
  infos = [
    device_mesh._DeviceStub(
      id=rank,
      coords=(rank, 0, 0),
      core_on_chip=0,
      device_kind="TPU v5e",
      platform="tpu",
      process_index=rank,
      slice_index=rank // local_count,
    )
    for rank in range(total)
  ]
  mesh_shape = (total,)
  dcn_shape = (slice_count,)
  device_mesh._validate_multi_slice(infos, mesh_shape, dcn_shape)
  assert math.prod(mesh_shape) == len(infos)


def test_get_device_assignment_returns_backend_rank_mapping(monkeypatch):
  device_ids = [7, 2, 5, 0]
  backend = object()
  tpu_device = object()
  original_device = device_mesh.torch.device
  process_group = types.SimpleNamespace(_get_backend=lambda device: backend)
  monkeypatch.setattr(device_mesh.dist, "is_initialized", lambda: True)
  monkeypatch.setattr(
    device_mesh.torch,
    "device",
    lambda device: tpu_device if device == "tpu" else original_device(device),
  )
  monkeypatch.setattr(
    device_mesh.tpu_distributed,
    "rank_to_device_id",
    lambda received_backend: device_ids if received_backend is backend else [],
  )
  assignment = device_mesh.get_device_assignment(process_group)
  assert assignment == device_ids
  assert sorted(assignment) == [0, 2, 5, 7]


@settings(max_examples=120, deadline=None)
@given(
  ranges=st.lists(
    st.tuples(
      st.integers(min_value=0, max_value=100),
      st.integers(min_value=0, max_value=100),
    ).filter(lambda pair: pair[0] <= pair[1]),
    min_size=1,
    max_size=5,
  )
)
def test_merge_bounds_contains_every_input_range(ranges):
  symbol = sympy.Symbol("s")
  input_bounds = [
    symbol_bounds.SymbolBounds({symbol: value_ranges.ValueRanges(lower, upper)})
    for lower, upper in ranges
  ]
  merged = symbol_bounds.SymbolBounds.merge_bounds(input_bounds)
  result = merged.symbol_ranges[symbol]
  assert int(result.lower) == min(lower for lower, _ in ranges)
  assert int(result.upper) == max(upper for _, upper in ranges)


@settings(max_examples=100, deadline=None)
@given(
  upper=st.integers(min_value=-100, max_value=100),
  constant=st.integers(min_value=-100, max_value=100),
  divisor=st.integers(min_value=1, max_value=20),
  remainder=st.integers(min_value=0, max_value=19),
)
def test_violated_guard_matches_direct_substitution(
  upper, constant, divisor, remainder
):
  assume(remainder < divisor)
  symbol = sympy.Symbol("s0", integer=True)
  expressions = (
    sympy.Eq(symbol, constant),
    sympy.Eq(sympy.Mod(symbol, divisor), remainder),
  )
  substitutions = {"s0": upper}
  for expression in expressions:
    env = types.SimpleNamespace(
      guards=[types.SimpleNamespace(expr=expression)], replacements={}
    )
    direct = expression.subs({symbol: upper})
    expected = None if direct == sympy.true else expression
    assert symbol_bounds.find_violated_guard(env, substitutions) == expected


def test_hardware_topology_sizes_match_device_counts():
  specs = (
    hardware.TpuSpec.V5P,
    hardware.TpuSpec.V5E,
    hardware.TpuSpec.V6E,
    hardware.TpuSpec.V7,
    hardware.TpuSpec.V8I,
    hardware.TpuSpec.V8T,
  )
  for spec in specs:
    for world_size, topology in spec.topologies.items():
      assert math.prod(topology) == world_size
      if spec.devices_per_chip > 1:
        assert topology[-1] == spec.devices_per_chip
        assert math.prod(topology[:-1]) * spec.devices_per_chip == world_size
