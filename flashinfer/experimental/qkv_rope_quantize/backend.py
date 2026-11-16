# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch


def quantize(
    q, k, v, cache, positions, qs, ks, vs, out, output_head_dim, backend, sin=None
):
    if backend != "triton":
        raise ValueError("backend must be triton")
    from .triton_backend import prepare

    if out is not None and len(out) != 3:
        raise ValueError("out must contain Q/K/V output tensors")
    if output_head_dim is None:
        output_head_dim = out[0].shape[-1] if out is not None else q.shape[-1]
    if not isinstance(output_head_dim, int):
        raise ValueError("output_head_dim must be an integer")
    if out is None:
        out = tuple(
            torch.empty(
                (*x.shape[:-1], output_head_dim),
                dtype=torch.float8_e4m3fn,
                device=q.device,
            )
            for x in (q, k, v)
        )
    elif any(x.shape[-1] != output_head_dim for x in out):
        raise ValueError("out feature dimensions must match output_head_dim")
    prepare(q, k, v, cache, positions, (qs, ks, vs), out, sin=sin)()
    return tuple(out)


def amax_partial(
    q, k, v, cache, positions, out, num_partials, reduction, backend, sin=None
):
    if backend != "triton":
        raise ValueError("backend must be triton")
    from .triton_backend import prepare_partials

    if reduction not in ("absmax", "max"):
        raise ValueError("reduction must be absmax or max")
    if num_partials is None:
        if out is not None and (out.ndim != 2 or out.shape[1] != 3):
            raise ValueError("out must have shape [num_partials, 3]")
        num_partials = out.shape[0] if out is not None else 1024
    if not isinstance(num_partials, int) or not 1 <= num_partials <= 1024:
        raise ValueError("num_partials must be an integer in [1, 1024]")
    if out is None:
        out = torch.empty((num_partials, 3), dtype=torch.float32, device=q.device)
    elif out.shape != (num_partials, 3):
        raise ValueError("out must have shape [num_partials, 3]")
    prepare_partials(q, k, v, cache, positions, out, reduction=reduction, sin=sin)()
    return out
