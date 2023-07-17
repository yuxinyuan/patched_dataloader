# PatchedDataLoader: Reproduce Pytorch DataLoader result with ``state_dict`` and ``load_state_dict``

## TL;DR

- ``PatchedDataLoader`` is a drop-in replacement of Pytorch DataLoader
- ``PatchedDataLoader`` automatically handles data loader seeding in single- and multi-GPUs setting
- ``PatchedDataLoader`` automatically snapshots the random states of DataLoader.
- One can call ``state_dict()`` and ``load_state_dict()`` to save/resume the data loading order.
- ``PatchedDataLoader`` automatically shards the sampler(map-style dataset) in distributed (multi-GPUs) training settings

PyTorch DataLoader provides a convenient way of reading map-style datasets. However, one cannot easily reproduce the data loading order of a DataLoader instance. I personally find it very frustrating when iterrupting and then resuming training, as I cannot restore the data loader state and the data loader has to start loading data from scratch.

``PatchedDataLoader`` inherits from PyTorch's ``DataLoader``. It attempts to address the randomness and reproducibility issues of DataLoader by seeding the generators for the DataLoader's generator and samplers, and automatically snapshoting these random states. It adds two commonly known methods: ``state_dict()`` and ``load_state_dict()`` to the original DataLoader. One can save the current state of the dataloader by calling ``state_dict()`` on an instance of PatchedDataLoader and load and restore this state by calling ``load_state_dict()``.

## Easy to use

To use ``PatchedDataLoader``, one just need to replace the original ``DataLoader`` with ``PatchedDataLoader``.

```diff
- from torch.utils.data import DataLoader
+ from patched_dataloader import  PatchedDataLoader as DataLoader

loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
```

You don't have to worry about whether you train with a single GPU or train with Pytorch DDP (multi-GPUs), ``PatchedDataLoader`` can be used as usual. To use ``state_dict`` and ``load_state_dict``, one can call them on a ``PatchedDataLoader`` instance just like calling ``state_dict`` and ``load_state_dict`` on ``torch.nn.Module`` instances:

```python
from patched_dataloader import  PatchedDataLoader as DataLoader

loader = DataLoader(list(range(15)), batch_size=5, shuffle=True)

for i, data in enumerate(loader):
    print(data)
    if i == 2:
        state = loader.state_dict()
        break

loader.load_state_dict(state)
for i, data in enumerate(loader):
    print(data)
```

## Additional benefits

Inspired by [accelerate](https://github.com/huggingface/accelerate), ``PatchedDataLoader`` patches the sampler of the data loader and shards the sampler (hence, the map-style dataset) across all distributed training processes. This means that the same code can be used for single-GPU and multi-GPUs training:

```diff
- import torch.distributed as dist
- from torch.utils.data import DataLoader, DistributedSampler
+ from patched_dataloader import  PatchedDataLoader as DataLoader

loader = DataLoader(train_dataste, batch_size=32, shuffle=True)
- if dist.is_initialized():
-     loader = DataLoader(train_dataset, batch_size=32, sampler=DistributedSampler(train_dataset))
```

## Limitations

After patching, the shuffle result of the dataset becomes deterministic, and we can thus recover the state of the DataLoader. However, when using multi-processing data loading, we cannot access and save the random state of worker processes. This means that we can never completely recover the state of a DataLoader (e.g. we cannot reproduce the results of data augmentation).
