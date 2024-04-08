"""Numba-compiled subroutines used for deep symbolic optimization."""

from numba import jit, prange
import numpy as np


# @jit(nopython=True, parallel=True)
def parents_siblings(tokens,  empty_parent, empty_sibling):
    """
    Given a batch of action sequences, computes and returns the parents and
    siblings of the next element of the sequence.

    The batch has shape (batch_size, sequence_length), where batch_size is the number of sequences (i.e. batch
    size) and sequence_length is the length of each sequence. In some cases, expressions may
    already be complete; in these cases, this function sees the start of a new
    expression, even though the return value for these elements won't matter
    because their gradients will be zero because of sequence_length.

    Parameters
    __________

    tokens : np.ndarray, shape=(batch_size, sequence_length), dtype=np.int32
        Batch of action sequences. Values correspond to library indices.

    arities : np.ndarray, dtype=np.int32
        Array of arities corresponding to library indices.

    parent_adjust : np.ndarray, dtype=np.int32
        Array of parent sub-library index corresponding to library indices.

    empty_parent : int
        Integer value for an empty parent token. This is initially computed in expression_decoder.py.

    empty_sibling : int
        Integer value for an empty sibling token. This is initially computed in expression_decoder.py

    Returns
    _______
    adj_parents : np.ndarray, shape=(batch_size,), dtype=np.int32
        Adjusted parents of the next element of each action sequence.

    siblings : np.ndarray, shape=(batch_size,), dtype=np.int32
        Siblings of the next element of each action sequence.

    """
    batch_size, sequence_length = tokens.shape

    adj_parents = np.full(shape=(batch_size,), fill_value=empty_parent, dtype=np.int32)
    siblings = np.full(shape=(batch_size,), fill_value=empty_sibling, dtype=np.int32)
    # Parallelized loop over action sequences
    for bi in range(batch_size):
        adj_parents[bi] = tokens[bi, -1]
        # siblings[bi] = tokens[bi, -1]
    return adj_parents, siblings




