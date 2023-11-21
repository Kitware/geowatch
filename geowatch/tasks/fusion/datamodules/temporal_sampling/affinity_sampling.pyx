#cython: language_level=3
"""
Faster affinity sampling (marginally)

References:
    https://github.com/numpy/numpy/issues/16686
    https://gist.github.com/kdubovikov/d4e5c688fa771227fdf8c924196a59fe
    https://gist.github.com/joshlk/5b1a2c3a8d4bcf94476cafa33e611795

Ignore:
    >>> import xdev
    >>> import pathlib
    >>> fpath = pathlib.Path('~/code/watch/watch/tasks/fusion/datamodules/affinity_sampling.pyx').expanduser()
    >>> renormalize_cython = xdev.import_module_from_pyx(fpath, recompile=True, verbose=3, annotate=True)
"""
import numpy as np
cimport numpy as cnp
cimport cython
# from numpy.random cimport bitgen_t
# from numpy.random import PCG64
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.time cimport time

## Need to seed random numbers with time otherwise will always get same results
srand(time(NULL))

# cdef inline long randint(long lower, long upper) nogil:
#     """Has the limitation that rand() only produces random integers upto RAND_MAX
#     which is guaranteed to be at least 32767 and usually 2,147,483,647 but this can vary on different systems.
#     Also the quality of random number aren't guaranteed and vary between systems"""
#     return rand() % (upper - lower + 1)


cdef inline double randfloat64() nogil:
    return (<double> rand()) / (<double> RAND_MAX)


@cython.boundscheck(False)
@cython.wraparound(False)
def cython_affinity_sample(cnp.ndarray affinity, int num_sample, cnp.ndarray current_weights, list chosen, rng=None) -> list:
    cdef cnp.ndarray available_idxs = np.arange(affinity.shape[0])
    cdef cnp.ndarray probs
    cdef cnp.ndarray cumprobs
    cdef cnp.ndarray next_affinity
    cdef int next_idx
    cdef int last_idx = len(current_weights) - 1

    # TODO: Can we speed this up?
    for _ in range(num_sample):
        # Choose the next image based on combined sample affinity

        cumprobs = current_weights.cumsum()

        # dart = rng.rand() * cumprobs[last_idx]
        # dart =  randfloat64() * cumprobs[last_idx]
        dart =  (<double> rand()) / (<double> RAND_MAX) * cumprobs[last_idx]

        next_idx = np.searchsorted(cumprobs, dart)
        
        # probs = current_weights / current_weights.sum()
        # next_idx = rng.choice(available_idxs, num_sample=1, p=probs)[0]

        next_affinity = affinity[next_idx]
        chosen.append(next_idx)

        # Don't resample the same item
        current_weights = current_weights * next_affinity
        # current_weights *= next_affinity
        current_weights[next_idx] = 0
    chosen = sorted(chosen)
    return chosen
