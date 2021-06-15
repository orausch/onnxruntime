import itertools
import collections
import argparse

from torch import nn, quantization as quant
import torch
from torch.utils import benchmark
from torch.nn import functional as F

import onnxruntime as rt

grid = dict(
    num_threads=[1, 2, 4, 12, 24],
    batch_size=[2 ** i for i in range(5, 17)],
    axis=(0, 1),
    feature_size=[64],
)

class Concatter(nn.Module):
    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def forward(self, x, y):
        return torch.cat([x, y], self.axis)



def grid_search(grid):
    """
    Generator for grid searching a grid.

    :param grid: the grid to search.
    :returns: iterable over dicts of Parameters.
    """

    T = collections.namedtuple("Parameters", grid.keys())

    return (T(**dict(zip(grid.keys(), v))) for v in itertools.product(*grid.values()))


def params_to_string(params) -> str:
    return ", ".join(f"{k}={getattr(params, k)}" for k in params._fields)


results = []
for params in grid_search(grid):

    module = Concatter(params.axis)

    a, b = torch.randn(params.batch_size, params.feature_size), torch.randn(params.batch_size, params.feature_size)
    torch.onnx.export(module, (a, b), "export.onnx", input_names=["x", "y"], output_names=["output"])


    sess_options = rt.SessionOptions()
    sess_options.intra_op_num_threads = params.num_threads

    sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

    ort_session = rt.InferenceSession("export.onnx", sess_options)


    batch = torch.rand(params.batch_size, 314)

    a_np = a.numpy()
    b_np = b.numpy()
    for i in range(5):
        ort_session.run(None, {"x": a_np, "y": b_np})

    def torch_func():
        module(a, b)

    def ort_func():
        output = ort_session.run(None, {"x": a_np, "y": b_np})[0]

    results.append(
        benchmark.Timer(
            stmt="func()",
            globals={"func": torch_func, "a": a, "b": b},
            num_threads=params.num_threads,
            label=f'axis={params.axis}',
            sub_label=params_to_string(params),
            description="torch",
        ).blocked_autorange(min_run_time=0.5)
    )

    results.append(
        benchmark.Timer(
            stmt="func()",
            globals={"func": ort_func, "a_np": a_np, "b_np": b_np},
            num_threads=params.num_threads,
            label=f'axis={params.axis}',
            sub_label=params_to_string(params),
            description="ort",
        ).blocked_autorange(min_run_time=0.5)
    )

compare = benchmark.Compare(results)

compare.colorize()
compare.print()
