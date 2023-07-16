import types
from typing import Protocol, runtime_checkable

import torch
from torch.utils.data import Sampler

from ._utils import get_unpatched_method, patch_unbound_method


@runtime_checkable
class _BatchSamplerProtocol(Protocol):
    batch_size: int
    drop_last: bool

    def __iter__(self):
        ...

    def __len__(self):
        ...


def is_batch_sampler(obj):
    return isinstance(obj, Sampler) and isinstance(obj, _BatchSamplerProtocol)


def sampler_build_generator(sampler: "Sampler", seed, tgt_attr="generator"):
    """Go through all the attr of `sampler`:

    - If the name of attr is in `tgt_attr` and the value of attr is None, we create
        a `torch.Generator` for it.

    - If the value of attr is a `Sampler`, we recursively go through the process again
    """
    tgt_attr = (tgt_attr,) if isinstance(tgt_attr, str) else tuple(tgt_attr)

    for k in sampler.__dict__:
        if k in tgt_attr:
            if sampler.__dict__[k] is None:
                sampler.__dict__[k] = torch.Generator().manual_seed(seed)
            elif isinstance(sampler.__dict__[k], torch.Generator):
                sampler.__dict__[k].manual_seed(seed)

        if isinstance(sampler.__dict__[k], Sampler):
            sampler_build_generator(sampler.__dict__[k], seed)


def sampler_state_dict(sampler: "Sampler"):
    ret = {}
    for k, v in sampler.__dict__.items():
        if isinstance(v, torch.Generator):
            ret[k] = v.get_state()
        if isinstance(v, Sampler):
            ret[k] = sampler_state_dict(v)
    return ret


def sampler_load_state_dict(sampler: "Sampler", state_dict):
    for k, v in state_dict.items():
        if k not in sampler.__dict__:
            # raise RuntimeError(f"key {k} of state dict doesn't match for {sampler}")
            continue
        if isinstance(sampler.__dict__[k], torch.Generator):
            sampler.__dict__[k].set_state(v)
        elif isinstance(sampler.__dict__[k], Sampler):
            sampler_load_state_dict(sampler.__dict__[k], v)


def shard_sampler__iter__(original_iter, batch_size, drop_last, rank, world_size):
    # Code adapted from https://github.com/huggingface/accelerate/blob/main/src/accelerate/data_loader.py
    # If `batch_size` is None, the sampler to shard is not a batch sampler, and the results yielded are
    # consider single element of index instead of list of indices.
    def ret(self, *args, **kwargs):
        initial_indices = []
        indices_to_yield = None
        for i, indices in enumerate(original_iter(self, *args, **kwargs)):
            if not drop_last and i < world_size:
                # Gather inital data, in case we need to add extra samples to
                # make batches evenly divsible to all ranks
                if batch_size is None:
                    initial_indices.append(indices)
                else:
                    initial_indices.extend(indices)

            if i % world_size == rank:
                indices_to_yield = indices
            if i % world_size == world_size - 1 and (batch_size is None or len(indices) == batch_size):
                # If the most recent batch `indices` is a full batch, we are safe
                # to yield indices for all ranks
                yield indices_to_yield
                indices_to_yield = None

        # In the case of not drop_last, it's possible that:
        # 1. some ranks get full batch of indices_to_yield
        # 2. one rank gets unfull batch of indices_to_yield
        # 3. some ranks get no indices_to_yield yet
        if not drop_last and len(initial_indices) > 0:
            if indices_to_yield is not None and (batch_size is None or len(indices_to_yield) == batch_size):
                # Case 1, for these ranks, just yield `indices_to_yield`
                yield indices_to_yield

            if batch_size is None or len(indices) == batch_size:
                # This means `indices` has been yielded in case 1.
                next_rank_to_yield = i % world_size + 1
                if next_rank_to_yield == world_size:
                    return
                remaining_indices = []
            else:
                # Case 2, `next_rank_to_yield` gets the last unfull batch of indices
                next_rank_to_yield, remaining_indices = i % world_size, indices
                if batch_size is None:
                    remaining_indices = [remaining_indices]

            # For case 3, go back to use `intial_data` to yield full batches for all ranks
            # We still need (world_size - next_rank_to_yield) * batch_size indices
            indices_needed = (
                (world_size - next_rank_to_yield) * batch_size
                if batch_size is not None
                else (world_size - next_rank_to_yield)
            )
            while len(remaining_indices) < indices_needed:
                remaining_indices += initial_indices

            if rank >= next_rank_to_yield:
                if batch_size is not None:
                    indices_to_yield = remaining_indices[
                        (rank - next_rank_to_yield) * batch_size : (rank - next_rank_to_yield + 1) * batch_size
                    ]
                    assert len(indices_to_yield) == batch_size
                    yield indices_to_yield
                else:
                    indices_to_yield = remaining_indices[(rank - next_rank_to_yield) : (rank - next_rank_to_yield + 1)]
                    yield indices_to_yield[0]

    return ret


def shard_sampler__len__(original_len, batch_size, drop_last, rank, world_size):
    # Code adapted from https://github.com/huggingface/accelerate/blob/main/src/accelerate/data_loader.py
    def ret(self, *args, **kwargs):
        length = original_len(self, *args, **kwargs)
        sharded_length = length // world_size

        if sharded_length * world_size == length:
            return sharded_length

        if drop_last:
            return sharded_length
        else:
            return sharded_length + 1

    return ret


def shard_sampler(sampler, batch_size, drop_last, rank, world_size):
    patch_unbound_method(sampler.__class__, "__iter__")
    patch_unbound_method(sampler.__class__, "__len__")

    sampler.__iter__ = types.MethodType(
        shard_sampler__iter__(get_unpatched_method(sampler, "__iter__"), batch_size, drop_last, rank, world_size),
        sampler,
    )
    sampler.__len__ = types.MethodType(
        shard_sampler__len__(get_unpatched_method(sampler, "__len__"), batch_size, drop_last, rank, world_size),
        sampler,
    )


def skip_sampler__iter__(original_iter):
    def ret(self, *args, **kwargs):
        _skip_num_iterations = getattr(self, "_skip_num_iterations", 0)

        if _skip_num_iterations > 0:
            for i, sample in enumerate(original_iter(self, *args, **kwargs)):
                if i >= _skip_num_iterations:
                    yield sample
            setattr(self, "_skip_num_iterations", 0)
        else:
            yield from original_iter(self, *args, **kwargs)

    return ret


def skip_sampler__len__(original_len):
    def ret(self, *args, **kwargs):
        _skip_num_iterations = getattr(self, "_skip_num_iterations", 0)
        return max(0, original_len(self, *args, **kwargs) - _skip_num_iterations)

    return ret


def skip_sampler(sampler, skip_num_iterations):
    patch_unbound_method(sampler.__class__, "__iter__")
    patch_unbound_method(sampler.__class__, "__len__")

    setattr(sampler, "_skip_num_iterations", skip_num_iterations)
    sampler.__iter__ = types.MethodType(skip_sampler__iter__(get_unpatched_method(sampler, "__iter__")), sampler)
    sampler.__len__ = types.MethodType(skip_sampler__len__(get_unpatched_method(sampler, "__len__")), sampler)
