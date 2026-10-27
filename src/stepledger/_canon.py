"""Canonical form and 64-bit digests of observations, actions and rewards.

A value is canonicalized by casting it to its declared space: every leaf of the
space (a Box, Discrete, MultiDiscrete or MultiBinary inside any nesting of Dict
and Tuple, dict keys sorted) becomes one little-endian numpy array. Both sides
of the boundary canonicalize with the same space, so equal values give equal
digests, and a cast to the declared dtype is never reported as a difference.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any, Sequence

import numpy as np
from gymnasium import spaces

KeyPath = tuple  # keys from the top-level value down to one leaf


def _leaf_dtype(space: spaces.Space) -> np.dtype:
    if isinstance(space, spaces.Discrete):
        dtype = np.dtype(np.int64)
    elif isinstance(space, (spaces.Box, spaces.MultiDiscrete, spaces.MultiBinary)):
        dtype = np.dtype(space.dtype)
    else:
        raise TypeError(
            f"stepledger does not support {type(space).__name__} spaces; supported are "
            "Box, Discrete, MultiDiscrete, MultiBinary, and Dict/Tuple of these"
        )
    return dtype.newbyteorder("<")


def _collect(space: spaces.Space, path: KeyPath, out: list[tuple[KeyPath, spaces.Space]]) -> None:
    if isinstance(space, spaces.Dict):
        for key in sorted(space.spaces, key=str):
            _collect(space.spaces[key], path + (key,), out)
    elif isinstance(space, spaces.Tuple):
        for i, sub in enumerate(space.spaces):
            _collect(sub, path + (i,), out)
    else:
        _leaf_dtype(space)  # raises for unsupported spaces
        out.append((path, space))


def describe(space: spaces.Space) -> str:
    """Describe what the canonical form of ``space`` depends on: types, shapes, dtypes, keys."""
    if isinstance(space, spaces.Dict):
        inner = ", ".join(f"{key!r}: {describe(space.spaces[key])}" for key in sorted(space.spaces, key=str))
        return f"Dict({inner})"
    if isinstance(space, spaces.Tuple):
        return "Tuple(" + ", ".join(describe(sub) for sub in space.spaces) + ")"
    return f"{type(space).__name__}(shape={space.shape}, dtype={_leaf_dtype(space).name})"


def path_name(path: KeyPath) -> str:
    """``('pos', 0)`` -> ``"['pos'][0]"``; the empty path is ``""``."""
    return "".join(f"[{key!r}]" for key in path)


class Canon:
    """Canonicalizes and digests values of one space."""

    def __init__(self, space: spaces.Space) -> None:
        leaves: list[tuple[KeyPath, spaces.Space]] = []
        _collect(space, (), leaves)
        self.space = space
        self.paths: list[KeyPath] = [path for path, _ in leaves]
        self.dtypes: list[np.dtype] = [_leaf_dtype(sub) for _, sub in leaves]
        self.names: list[str] = [path_name(path) for path in self.paths]
        self.description = describe(space)
        self._single = self.paths == [()]
        self._headers: list[dict[tuple, bytes]] = [{} for _ in self.paths]  # per leaf: shape -> header

    def leaves(self, value: Any) -> list[np.ndarray]:
        """Canonical leaves of one value.

        Also works on a batched value (gymnasium and SB3 batch Dict/Tuple values
        leaf by leaf); then row ``i`` of each leaf belongs to env ``i``.
        """
        if self._single:
            return [np.asarray(value, dtype=self.dtypes[0])]
        out = []
        for path, dtype in zip(self.paths, self.dtypes):
            item = value
            for key in path:
                item = item[key]
            out.append(np.asarray(item, dtype=dtype))
        return out

    def digest(self, leaves: Sequence[np.ndarray]) -> int:
        """64-bit blake2b digest of the canonical leaves.

        For each leaf, in order, the hash is fed its path name (utf-8), its
        ndim (uint8) and shape (int64 each, little-endian), its dtype string
        (e.g. ``<f4``) and its bytes in C order.
        """
        h = hashlib.blake2b(digest_size=8)
        for j, leaf in enumerate(leaves):
            header = self._headers[j].get(leaf.shape)
            if header is None:  # the dtype is fixed per leaf, so the header only depends on the shape
                header = self.names[j].encode() + struct.pack(f"<B{leaf.ndim}q", leaf.ndim, *leaf.shape)
                header += leaf.dtype.str.encode()
                self._headers[j][leaf.shape] = header
            h.update(header)
            h.update(leaf if leaf.flags.c_contiguous else np.ascontiguousarray(leaf))
        return int.from_bytes(h.digest(), "little")


_UINT_OF_SIZE = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}


def reward_dtype_of(reward_dtype: Any) -> np.dtype:
    """Validate a reward dtype: a numeric dtype of at most 8 bytes."""
    dtype = np.dtype(reward_dtype)
    if dtype.kind not in "fiub" or dtype.itemsize not in _UINT_OF_SIZE:
        raise ValueError(f"reward_dtype must be a numeric dtype of 1, 2, 4 or 8 bytes, got {dtype}")
    return dtype.newbyteorder("<")


def reward_bits(reward: Any, dtype: np.dtype) -> int:
    """The reward cast to ``dtype``, as its raw bits (an exact, reversible stand-in for a digest)."""
    value = np.array(reward, dtype=dtype).reshape(())
    return int(value.view(_UINT_OF_SIZE[dtype.itemsize]))


def reward_bits_batch(rewards: Any, dtype: np.dtype) -> list[int]:
    """:func:`reward_bits` for a batch of rewards."""
    values = np.asarray(rewards, dtype=dtype).reshape(-1)
    return values.view(_UINT_OF_SIZE[dtype.itemsize]).tolist()


def reward_from_bits(bits: int, dtype: np.dtype) -> np.generic:
    """Inverse of :func:`reward_bits`."""
    return np.asarray(bits, dtype=_UINT_OF_SIZE[dtype.itemsize]).view(dtype)[()]
