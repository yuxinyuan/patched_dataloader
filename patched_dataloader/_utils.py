import random
import types

import numpy
import torch


def set_random_seed(seed, worker=0):
    # Code taken from https://github.com/pytorch/pytorch/blob/main/torch/utils/data/_utils/worker.py
    def _generate_state(base_seed, worker_id):
        INIT_A = 0x43B0D7E5
        MULT_A = 0x931E8875
        INIT_B = 0x8B51F9DD
        MULT_B = 0x58F38DED
        MIX_MULT_L = 0xCA01F9DD
        MIX_MULT_R = 0x4973F715
        XSHIFT = 4 * 8 // 2
        MASK32 = 0xFFFFFFFF

        entropy = [worker_id, base_seed & MASK32, base_seed >> 32, 0]
        pool = [0] * 4

        hash_const_A = INIT_A

        def hash(value):
            nonlocal hash_const_A
            value = (value ^ hash_const_A) & MASK32
            hash_const_A = (hash_const_A * MULT_A) & MASK32
            value = (value * hash_const_A) & MASK32
            value = (value ^ (value >> XSHIFT)) & MASK32
            return value

        def mix(x, y):
            result_x = (MIX_MULT_L * x) & MASK32
            result_y = (MIX_MULT_R * y) & MASK32
            result = (result_x - result_y) & MASK32
            result = (result ^ (result >> XSHIFT)) & MASK32
            return result

        # Add in the entropy to the pool.
        for i in range(len(pool)):
            pool[i] = hash(entropy[i])

        # Mix all bits together so late bits can affect earlier bits.
        for i_src in range(len(pool)):
            for i_dst in range(len(pool)):
                if i_src != i_dst:
                    pool[i_dst] = mix(pool[i_dst], hash(pool[i_src]))

        hash_const_B = INIT_B
        state = []
        for i_dst in range(4):
            data_val = pool[i_dst]
            data_val = (data_val ^ hash_const_B) & MASK32
            hash_const_B = (hash_const_B * MULT_B) & MASK32
            data_val = (data_val * hash_const_B) & MASK32
            data_val = (data_val ^ (data_val >> XSHIFT)) & MASK32
            state.append(data_val)
        return state

    final_seed = seed + worker
    random.seed(final_seed)
    torch.manual_seed(final_seed)

    np_seed = _generate_state(seed, worker)  # Numpy only accepts 32bit seed
    numpy.random.seed(np_seed)


def patch_unbound_method(cls, meth):
    unpatched_method = getattr(cls, meth)
    if hasattr(unpatched_method, "_patched") and getattr(unpatched_method, "_patched"):
        return

    def patch_func(self, *args, **kwargs):
        """Use this function to patch unbound dunder methods so that, when we patch
        bound methods, we can invoke the patched bound method with python built-in functions
        (e.g. call iter to invoke __iter__)
        """
        if meth in self.__dict__:
            return self.__dict__[meth](*args, **kwargs)
        # Note that unpatched_method is unbound, we need to pass in self.
        return unpatched_method(self, *args, **kwargs)

    patch_func._patched = True
    setattr(cls, meth, patch_func)
    setattr(cls, "_original_" + meth, unpatched_method)


def get_unpatched_method(obj, method_name):
    if method_name in obj.__dict__:
        method = obj.__dict__[method_name]
        if isinstance(method, types.MethodType):
            # Turn a bound method into unbound function
            method = method.__func__
    else:
        method = getattr(obj.__class__, method_name)

    if hasattr(method, "_patched") and getattr(method, "_patched"):
        return getattr(obj.__class__, "_original_" + method_name)
    return method
