import inspect

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset, Sampler, get_worker_info

from ._utils import set_random_seed
from .sampler import (
    is_batch_sampler,
    sampler_build_generator,
    sampler_load_state_dict,
    sampler_state_dict,
    shard_sampler,
    skip_sampler,
)


class ReSeedDataLoaderWorker:
    def __init__(self, worker_init_fn=None, world_size=1, rank=0):
        self.worker_init_fn = worker_init_fn
        self.world_size = world_size
        self.rank = rank

    def __call__(self, id):
        worker_info = get_worker_info()
        if worker_info is not None:
            # By default, DataLoader worker sets seed as base_seed + worker_id
            # In the case of DDP training, if all ranks share the same base_seed,
            # then rank 0 worker 0 will have identical seed as rank1 worker 0.
            # We re-seed the workers to have seed:
            # base_seed + worker_id + num_workers * rank
            set_random_seed(worker_info.seed + worker_info.num_workers * self.rank)

        if self.worker_init_fn is not None:
            self.worker_init_fn(id)


class _PatchedDataLoaderBase(DataLoader):
    def __init_subclass__(cls):
        super().__init_subclass__()

        original_init = cls.__init__
        default_post_init_kwargs = _PatchedDataLoaderBase._get_default_kwargs_from_meth_signature(cls.post_init)

        def patched_init(self: "_PatchedDataLoaderBase", *args, **kwargs):
            post_init_kwargs = {}
            for k, v in default_post_init_kwargs.items():
                if v == inspect.Parameter.empty and k not in kwargs:
                    raise RuntimeError(f"Missing value for argument `{k}` in post_init().")
                post_init_kwargs[k] = kwargs.pop(k, v)

            original_init(self, *args, **kwargs)
            self.post_init(**post_init_kwargs)

        cls.__init__ = patched_init

    @staticmethod
    def _get_default_kwargs_from_meth_signature(meth, *, exclude_keys=("self",)):
        signature = inspect.signature(meth)
        params = signature.parameters
        ret = {}
        for k, v in params.items():
            if k in exclude_keys:
                continue
            if v.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD:
                raise RuntimeError(
                    f"Parameter {v} of {meth.__qualname__} {signature} has kind {v.kind}, "
                    f"but we expect kind {inspect.Parameter.POSITIONAL_OR_KEYWORD} only"
                )
            ret[k] = v.default
        return ret

    def post_init(self, *args, **kwargs):
        pass


class PatchedDataLoader(_PatchedDataLoaderBase):
    def post_init(self, seed=1024, shard_samplers=True):
        self.seed = seed
        self._shard_samplers = shard_samplers
        self._original_worker_init_fn = self.worker_init_fn
        self.__last_random_state = None
        self.__iterator_ref = None

        # Build a generator if `self.generator` is None.
        # Otherwise, re-build a generator with the same state.
        # This help us separate DataLoader's generator from Sampler's generator
        self.generator = (
            torch.Generator().manual_seed(seed)
            if self.generator is None
            else torch.Generator().set_state(self.generator.get_state())
        )

        # Build generator for samplers, this helps us re-produce the shuffle
        # result of DataLoader.
        if isinstance(self.batch_sampler, Sampler):
            sampler_build_generator(self.batch_sampler, seed)
        if isinstance(self.sampler, Sampler):
            sampler_build_generator(self.sampler, seed)

        self.world_size = 1
        self.rank = 0
        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()

        self.shard_samplers()
        self.patch_worker_init_fn()

    def snapshot(self):
        """This function should be called before DataLoader consumes any generator states
        so that, when we resume data loading, we can resume DataLoader's random state and,
        re-play the exact data loading order we left last time.

        In practice, we call snapshot before and after __iter__ call.
        """
        self.__last_random_state = {
            "sampler": sampler_state_dict(self.sampler) if isinstance(self.sampler, Sampler) else {},
            "batch_sampler": sampler_state_dict(self.batch_sampler) if isinstance(self.batch_sampler, Sampler) else {},
            "generator": self.generator.get_state(),
        }

    def state_dict(self):
        """This function will dump any necessary state to restore the current data loading
        order of the dataloader. One can use `load_state_dict` to resume a dataloader.
        """
        # If `self.__iterator_ref` is not None, we are in the middle of data loading. In this case, we
        # record the number of iterations yielded so far to restore the current state of the DataLoader
        # as accurately as possible in the future.
        num_iterations_yielded = 0 if self.__iterator_ref is None else self.__iterator_ref._num_yielded
        _, batch_size, _ = self._get_batch_info()
        if batch_size is None:
            batch_size = 1
        # Actual number of data samples yielded by the dataloader.
        num_samples_yielded = num_iterations_yielded * batch_size * self.world_size
        return {
            "random_state": self.__last_random_state,
            "num_samples_yielded": num_samples_yielded,
        }

    def load_state_dict(self, state_dict, proceed=True):
        """This function will load `state_dict` into the current dataloader, and resume the data loading
        order by: 1. loading generator states saved in `state_dict`; 2. patching samplers to skip the first
        `num_samples_yielded` saved in `state_dict`

        If `proceed` is True, the dataloader continues to load data from where it left off last time.
        If `proceed` is False, the dataloader reload the last data loaded when it was shutdown. This
        should be useful for debug purpose.
        """
        if state_dict is None:
            return
        self._iterator = None
        random_state = state_dict["random_state"]
        num_samples_yielded = state_dict["num_samples_yielded"]

        if random_state is not None:
            if isinstance(self.sampler, Sampler):
                sampler_load_state_dict(self.sampler, random_state["sampler"])
            if isinstance(self.batch_sampler, Sampler):
                sampler_load_state_dict(self.batch_sampler, random_state["batch_sampler"])
            self.generator.set_state(random_state["generator"])

        self.skip_samplers(num_samples_yielded, proceed)

    def skip_samplers(self, skip_num_samples, proceed):
        if not isinstance(self.dataset, IterableDataset) and skip_num_samples > 0:
            sampler_to_skip, batch_size, _ = self._get_batch_info()

            if batch_size is None:
                batch_size = 1
            skip_num_iterations = skip_num_samples // batch_size // self.world_size
            if not proceed:
                skip_num_iterations = max(0, skip_num_iterations - 1)
            skip_sampler(sampler_to_skip, skip_num_iterations)

    def _get_batch_info(self):
        if is_batch_sampler(self.sampler):
            # It's rare to meet this case, but we handle it anyway
            # We assume `self.batch_sampler` is either None or the default BatchSampler
            # with batch size of 1.
            batch_size = getattr(self.sampler, "batch_size", self.batch_size)
            drop_last = getattr(self.sampler, "drop_last", self.drop_last)
            sampler = self.sampler
        elif self.batch_sampler is not None:
            batch_size = getattr(self.batch_sampler, "batch_size", self.batch_size)
            drop_last = getattr(self.batch_sampler, "drop_last", self.drop_last)
            sampler = self.batch_sampler
        else:
            batch_size = None
            drop_last = self.drop_last
            sampler = self.sampler
        return sampler, batch_size, drop_last

    def shard_samplers(self):
        if self._shard_samplers and not isinstance(self.dataset, IterableDataset) and self.world_size > 1:
            sampler_to_shard, batch_size, drop_last = self._get_batch_info()
            shard_sampler(sampler_to_shard, batch_size, drop_last, self.rank, self.world_size)

    def patch_worker_init_fn(self):
        self.worker_init_fn = ReSeedDataLoaderWorker(
            self._original_worker_init_fn,
            world_size=self.world_size,
            rank=self.rank,
        )

    def __iter__(self):
        # We turn self.__iter__ into a generator so that we know when we
        # will reach the end of the iterator
        self.snapshot()
        self.__iterator_ref = super().__iter__()
        yield from self.__iterator_ref
        # Take a snapshot, because the last iteration has finished and we
        # will never need the stale state again
        self.snapshot()
        self.__iterator_ref = None
