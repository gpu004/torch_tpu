"""Property tests for dynamic view decomposition on CPU tensors."""

from __future__ import annotations

import math

from hypothesis import HealthCheck, assume, given, settings, strategies as st
import pytest
import torch

from conftest import EXPRESSIBLE_VIEW_NOT_IMPLEMENTED
from torch_tpu._internal.compile.dynamic import view_decomposition as vd


@st.composite
def _view_cases(draw):
  rank = draw(st.integers(min_value=1, max_value=4))
  shape = tuple(
    draw(st.integers(min_value=0, max_value=6)) for _ in range(rank)
  )
  base = torch.arange(math.prod(shape), dtype=torch.int64).reshape(shape)
  tensor = base
  ops = draw(
    st.lists(
      st.sampled_from([
        "narrow",
        "slice1",
        "slice_step",
        "permute",
        "transpose",
        "unsqueeze",
        "select",
        "expand",
        "t",
        "reshape",
        "view",
        "diagonal",
        "as_strided",
      ]),
      min_size=1,
      max_size=4,
    )
  )
  simple = True
  for op in ops:
    rank = tensor.dim()
    if op in ("narrow", "slice1", "slice_step") and rank:
      dim = draw(st.integers(min_value=0, max_value=rank - 1))
      length = tensor.size(dim)
      start = draw(st.integers(min_value=0, max_value=length))
      end = draw(st.integers(min_value=start, max_value=length))
      step = (
        1 if op != "slice_step" else draw(st.integers(min_value=2, max_value=4))
      )
      if op == "narrow":
        size = end - start
        tensor = tensor.narrow(dim, start, size)
      else:
        tensor = tensor[
          tuple(
            slice(start, end, step) if i == dim else slice(None)
            for i in range(rank)
          )
        ]
      simple &= step == 1
    elif op == "permute" and rank >= 2:
      permutation = draw(st.permutations(tuple(range(rank))))
      tensor = tensor.permute(permutation)
      simple = False
    elif op == "transpose" and rank >= 2:
      dim0 = draw(st.integers(min_value=0, max_value=rank - 1))
      dim1 = draw(st.integers(min_value=0, max_value=rank - 1))
      tensor = tensor.transpose(dim0, dim1)
      simple = False
    elif op == "unsqueeze" and rank < 6:
      dim = draw(st.integers(min_value=0, max_value=rank))
      tensor = tensor.unsqueeze(dim)
      simple = False
    elif op == "select" and rank and tensor.size(0) > 0:
      dim = draw(st.integers(min_value=0, max_value=rank - 1))
      if tensor.size(dim):
        index = draw(st.integers(min_value=0, max_value=tensor.size(dim) - 1))
        tensor = tensor.select(dim, index)
        simple = False
    elif op == "expand" and rank < 5:
      leading = draw(st.integers(min_value=0, max_value=min(2, 5 - rank)))
      prefix = tuple(
        draw(st.integers(min_value=1, max_value=3)) for _ in range(leading)
      )
      target = tuple(
        0
        if size == 0
        else (
          draw(st.integers(min_value=0, max_value=4)) if size == 1 else size
        )
        for size in tensor.shape
      )
      tensor = tensor.expand(prefix + target)
      simple = False
    elif op == "t" and rank == 2:
      tensor = tensor.t()
      simple = False
    elif op == "reshape":
      tensor = tensor.reshape(-1)
      simple = False
    elif op == "view":
      try:
        tensor = tensor.view(-1)
        simple = False
      except RuntimeError:
        pass
    elif op == "diagonal" and rank >= 2:
      dim1 = draw(st.integers(min_value=0, max_value=rank - 1))
      dim2 = draw(st.integers(min_value=0, max_value=rank - 1))
      if dim1 != dim2:
        offset = draw(
          st.integers(min_value=-tensor.size(dim1), max_value=tensor.size(dim2))
        )
        tensor = tensor.diagonal(offset=offset, dim1=dim1, dim2=dim2)
        simple = False
    elif op == "as_strided":
      storage_numel = tensor.untyped_storage().nbytes() // tensor.element_size()
      new_rank = draw(st.integers(min_value=1, max_value=4))
      new_shape = tuple(
        draw(st.integers(min_value=0, max_value=4)) for _ in range(new_rank)
      )
      strides = tuple(
        draw(st.integers(min_value=0, max_value=storage_numel + 1))
        for _ in range(new_rank)
      )
      max_offset = sum(
        (dim - 1) * stride for dim, stride in zip(new_shape, strides) if dim > 0
      )
      if math.prod(new_shape) == 0:
        offset = 0
      elif max_offset < storage_numel:
        offset = draw(
          st.integers(min_value=0, max_value=storage_numel - 1 - max_offset)
        )
      else:
        new_shape = (0,)
        strides = (1,)
        offset = 0
      tensor = torch.as_strided(tensor, new_shape, strides, offset)
      simple = False
  return base, tensor, tuple(ops), simple


def _apply_ops(tensor, base_shape, ops):
  base_strides = [1] * len(base_shape)
  for dim in reversed(range(len(base_shape) - 1)):
    base_strides[dim] = base_strides[dim + 1] * base_shape[dim + 1]
  result = torch.as_strided(
    tensor,
    size=tuple(base_shape),
    stride=tuple(base_strides),
    storage_offset=0,
  )
  for kind, args in ops:
    if kind == "slice":
      dim, start, end, step = args
      slices = [slice(None)] * result.dim()
      slices[dim] = slice(start, end, step)
      result = result[tuple(slices)]
    elif kind == "permute":
      (permutation,) = args
      result = result.permute(permutation)
    else:
      raise AssertionError(f"Unexpected decomposition operation: {kind}")
  return result


@settings(
  max_examples=250,
  deadline=None,
  suppress_health_check=[HealthCheck.too_slow],
)
@given(case=_view_cases())
def test_view_decomposition_round_trips(case):
  _, tensor, chain, simple_chain = case
  assume(tensor.dim() != 0 or tensor.storage_offset() == 0)
  try:
    decomposition = vd.decompose_into_view_sequence(tensor)
  except NotImplementedError:
    if simple_chain:
      EXPRESSIBLE_VIEW_NOT_IMPLEMENTED.append((
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
      ))
    try:
      vd.get_base_tensor(tensor)
    except NotImplementedError:
      return
    except Exception as exc:
      pytest.fail(
        f"get_base_tensor raised {type(exc).__name__} for "
        f"shape={tuple(tensor.shape)}, stride={tensor.stride()}, chain={chain}:"
        f" {exc}"
      )
    pytest.fail(
      "get_base_tensor succeeded although decomposition raised "
      f"NotImplementedError for shape={tuple(tensor.shape)}, chain={chain}"
    )
  except Exception as exc:
    pytest.fail(
      f"decompose_into_view_sequence raised {type(exc).__name__} for "
      f"shape={tuple(tensor.shape)}, stride={tensor.stride()}, chain={chain}:"
      f" {exc}"
    )

  assert (decomposition is None) == vd._is_contiguous_tensor(tensor)
  storage_numel = tensor.untyped_storage().nbytes() // tensor.element_size()
  if decomposition is not None:
    base_shape, view_ops = decomposition
    assume(math.prod(base_shape) <= storage_numel)

  try:
    base_tensor = vd.get_base_tensor(tensor)
  except NotImplementedError:
    return
  except Exception as exc:
    pytest.fail(
      f"get_base_tensor raised {type(exc).__name__} for chain={chain}: {exc}"
    )

  assert math.prod(base_tensor.shape) <= storage_numel
  if decomposition is None:
    assert tuple(base_tensor.shape) == tuple(tensor.shape)
    return

  assert math.prod(base_shape) <= storage_numel
  reconstructed = _apply_ops(tensor, base_shape, view_ops)
  assert tuple(reconstructed.shape) == tuple(tensor.shape)
  assert all(
    actual == expected
    for size, actual, expected in zip(
      tensor.shape, reconstructed.stride(), tensor.stride()
    )
    if size > 1
  )
  assert reconstructed.storage_offset() == tensor.storage_offset()
  torch.testing.assert_close(reconstructed, tensor)
  assert tuple(base_tensor.shape) == tuple(base_shape)


@pytest.mark.xfail(
  strict=True, raises=IndexError, reason="google-pytorch/torch_tpu#5671"
)
def test_scalar_view_with_nonzero_storage_offset_decomposes():
  tensor = torch.tensor([0, 1]).select(0, 1)
  vd.decompose_into_view_sequence(tensor)


@pytest.mark.xfail(
  strict=True, raises=RuntimeError, reason="google-pytorch/torch_tpu#5672"
)
def test_unit_dimension_view_base_fits_storage():
  tensor = torch.arange(9.0)[::2, None]
  base_tensor = vd.get_base_tensor(tensor)
  decomposition = vd.decompose_into_view_sequence(tensor)
  storage_numel = tensor.untyped_storage().nbytes() // tensor.element_size()

  assert decomposition is not None
  assert math.prod(decomposition[0]) <= storage_numel
  assert math.prod(base_tensor.shape) <= storage_numel


def _hint_value(value):
  if isinstance(value, torch.SymInt):
    return value.node.hint
  if isinstance(value, (tuple, list)):
    return tuple(_hint_value(item) for item in value)
  return value


def _has_symbolic_layout(tensor):
  return any(
    isinstance(value, torch.SymInt)
    for value in (*tensor.shape, *tensor.stride(), tensor.storage_offset())
  )


@pytest.mark.parametrize(
  ("case_name", "make_view"),
  [
    pytest.param("x[:, 1:]", lambda x: x[:, 1:], id="column-slice"),
    pytest.param("x.t()", lambda x: x.t(), id="transpose"),
    pytest.param("x[1]", lambda x: x[1], id="row-select"),
    pytest.param(
      "x[:, 1:3].t()", lambda x: x[:, 1:3].t(), id="slice-transpose"
    ),
    pytest.param(
      "x.permute(1, 0)[1:]",
      lambda x: x.permute(1, 0)[1:],
      id="permuted-slice",
    ),
    pytest.param(
      "x[1:]",
      lambda x: x[1:],
      marks=pytest.mark.xfail(
        strict=True, reason="google-pytorch/torch_tpu#5674"
      ),
      id="row-slice",
    ),
    pytest.param(
      "x[1:, None]",
      lambda x: x[1:, None],
      marks=pytest.mark.xfail(
        strict=True, reason="google-pytorch/torch_tpu#5674"
      ),
      id="row-slice-unsqueeze",
    ),
    pytest.param(
      "x.unsqueeze(1)[2:]",
      lambda x: x.unsqueeze(1)[2:],
      marks=pytest.mark.xfail(
        strict=True, reason="google-pytorch/torch_tpu#5674"
      ),
      id="unsqueeze-row-slice",
    ),
    pytest.param(
      "x[None, 2:]",
      lambda x: x[None, 2:],
      marks=pytest.mark.xfail(
        strict=True, reason="google-pytorch/torch_tpu#5674"
      ),
      id="leading-unsqueeze-slice",
    ),
    pytest.param(
      "x[2:, :1]",
      lambda x: x[2:, :1],
      marks=pytest.mark.xfail(
        strict=True, reason="google-pytorch/torch_tpu#5674"
      ),
      id="row-and-unit-column-slice",
    ),
  ],
)
def test_dynamo_view_placeholder_decomposition_matches_concrete(
  case_name, make_view, monkeypatch
):
  del case_name
  torch._dynamo.reset()
  monkeypatch.setattr(torch._dynamo.config, "automatic_dynamic_shapes", True)
  symbolic_placeholders = []

  def backend(graph_module, example_inputs):
    del example_inputs
    for node in graph_module.graph.find_nodes(op="placeholder"):
      tensor = node.meta.get("example_value")
      if isinstance(tensor, torch.Tensor):
        symbolic_placeholders.append((
          tensor,
          vd.decompose_into_view_sequence(tensor),
        ))
    return graph_module.forward

  compiled = torch.compile(lambda x: x * 2, backend=backend)
  second_input = None
  for shape in ((5, 6), (7, 8)):
    base = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
    tensor = make_view(base)
    compiled(tensor)
    second_input = tensor

  symbolic_decompositions = [
    decomposition
    for tensor, decomposition in symbolic_placeholders
    if _has_symbolic_layout(tensor)
  ]
  assert symbolic_decompositions

  concrete_decomposition = vd.decompose_into_view_sequence(second_input)
  concrete_base_shape = (
    tuple(concrete_decomposition[0])
    if concrete_decomposition is not None
    else None
  )
  for decomposition in symbolic_decompositions:
    symbolic_base_shape = (
      tuple(_hint_value(dimension) for dimension in decomposition[0])
      if decomposition is not None
      else None
    )
    assert symbolic_base_shape == concrete_base_shape


@settings(max_examples=100, deadline=None)
@given(lower=st.integers(min_value=0, max_value=10**6))
def test_fallback_upper_bound_is_not_below_shape_lower_bound(lower):
  from torch_tpu._internal.compile.dynamic import symbol_bounds

  assert symbol_bounds.get_fallback_upper_bound(lower) >= lower


def test_symbolic_transpose_matches_concrete_decomposition():
  torch._dynamo.reset()
  captured = {}

  def backend(graph_module, example_inputs):
    del example_inputs
    for node in graph_module.graph.nodes:
      if node.op in ("call_function", "call_method") and any(
        operation in str(node.target) for operation in ("transpose", "permute")
      ):
        captured["node"] = node
        break
    return graph_module.forward

  @torch.compile(backend=backend, dynamic=True)
  def transpose_and_add(tensor):
    view = tensor.transpose(1, 2)
    return view + view

  base = torch.arange(24, dtype=torch.int64).reshape(2, 3, 4)
  torch._dynamo.mark_dynamic(base, 1, min=2, max=8)
  transpose_and_add(base)

  node = captured.get("node")
  if node is None:
    pytest.skip("TorchDynamo graph did not expose a transpose/permute node")
  if "val" not in node.meta:
    node.meta["val"] = node.meta.get("example_value")
  if node.meta.get("val") is None:
    pytest.skip("TorchDynamo did not attach a symbolic tensor value")
  captured["decomposition"] = vd.decompose_into_view_sequence(node)
  symbolic_base_shape, symbolic_ops = captured["decomposition"]
  concrete = base.transpose(1, 2)
  concrete_decomposition = vd.decompose_into_view_sequence(concrete)
  assert concrete_decomposition is not None
  concrete_base_shape, concrete_ops = concrete_decomposition

  assert tuple(map(int, symbolic_base_shape)) == tuple(concrete_base_shape)
  normalize = lambda ops: [
    (kind, (tuple(map(int, args[0])),) if kind == "permute" else args)
    for kind, args in ops
  ]
  assert normalize(symbolic_ops) == normalize(concrete_ops)
  assert tuple(map(int, node.meta["val"].shape)) == tuple(concrete.shape)
