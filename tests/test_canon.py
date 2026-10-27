"""Canonical values: the same value digests the same on both sides, a changed one does not."""

import hashlib
import struct

import numpy as np
import pytest
from gymnasium import spaces

from stepledger._canon import Canon, describe, reward_bits, reward_dtype_of, reward_from_bits


def digest(space, value):
    canon = Canon(space)
    return canon.digest(canon.leaves(value))


def test_box_values_are_cast_to_the_space_dtype():
    box = spaces.Box(-1, 1, (3,), np.float32)
    value = np.array([0.1, -0.2, 0.3])
    assert digest(box, value) == digest(box, value.astype(np.float32)) == digest(box, value.tolist())
    assert digest(box, value) != digest(box, value.astype(np.float16))
    assert digest(box, value) != digest(box, value[:2])


def test_shape_is_part_of_the_digest():
    box = spaces.Box(-1, 1, (4,), np.float32)
    flat = np.zeros(4, dtype=np.float32)
    assert digest(box, flat) != digest(box, flat.reshape(2, 2))


def test_discrete_values_are_int64():
    space = spaces.Discrete(4)
    assert digest(space, 2) == digest(space, np.int64(2)) == digest(space, np.int32(2)) == digest(space, np.array(2))
    assert digest(space, 2) != digest(space, 3)


def test_multi_discrete_and_multi_binary_use_the_space_dtype():
    md = spaces.MultiDiscrete([3, 4])
    assert digest(md, [1, 2]) == digest(md, np.array([1, 2], dtype=np.int32))
    mb = spaces.MultiBinary(3)
    assert Canon(mb).dtypes == [np.dtype(np.int8)]
    assert digest(mb, [1, 0, 1]) != digest(mb, [1, 1, 1])


def test_dict_keys_are_sorted_and_tuples_nest():
    space = spaces.Dict(
        {"b": spaces.Discrete(2), "a": spaces.Tuple((spaces.Box(0, 1, (2,)), spaces.Discrete(3)))}
    )
    canon = Canon(space)
    assert canon.names == ["['a'][0]", "['a'][1]", "['b']"]
    one = {"b": 1, "a": (np.array([0.5, 0.25]), 2)}
    other = {"a": (np.array([0.5, 0.25]), 2), "b": 1}
    assert digest(space, one) == digest(space, other)
    assert digest(space, one) != digest(space, {"b": 1, "a": (np.array([0.5, 0.25]), 1)})
    assert describe(space) == (
        "Dict('a': Tuple(Box(shape=(2,), dtype=float32), Discrete(shape=(), dtype=int64)), "
        "'b': Discrete(shape=(), dtype=int64))"
    )


def test_batched_leaves_give_the_same_digest_per_row():
    space = spaces.Dict({"x": spaces.Box(-1, 1, (2,)), "n": spaces.Discrete(5)})
    canon = Canon(space)
    batch = {"x": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32), "n": np.array([3, 4])}
    leaves = canon.leaves(batch)
    for i in range(2):
        single = {"x": batch["x"][i], "n": int(batch["n"][i])}
        assert canon.digest([leaf[i] for leaf in leaves]) == digest(space, single)


def test_digest_is_the_documented_hash():
    # name + ndim (uint8) + shape (int64 LE) + dtype string + C-order bytes, per leaf
    space = spaces.Dict({"b": spaces.Discrete(3), "a": spaces.Box(0, 1, (2, 2), np.float32)})
    a = np.arange(4, dtype=np.float32).reshape(2, 2)
    h = hashlib.blake2b(digest_size=8)
    h.update(b"['a']" + struct.pack("<B2q", 2, 2, 2) + b"<f4" + a.tobytes())
    h.update(b"['b']" + struct.pack("<B", 0) + b"<i8" + np.int64(2).tobytes())
    expected = int.from_bytes(h.digest(), "little")
    assert digest(space, {"a": a, "b": 2}) == expected
    assert digest(space, {"a": np.asfortranarray(a), "b": 2}) == expected  # memory layout does not matter
    assert digest(space, {"a": a.T.copy().T, "b": 2}) == expected


def test_unsupported_space_is_refused():
    with pytest.raises(TypeError, match="Text"):
        Canon(spaces.Text(5))


def test_reward_bits_are_exact_and_reversible():
    f32 = reward_dtype_of(np.float32)
    assert reward_bits(0.1, f32) == reward_bits(np.float32(0.1), f32)
    assert reward_from_bits(reward_bits(-0.731, f32), f32) == np.float32(-0.731)
    assert reward_bits(-0.731, f32) != reward_bits(np.float16(-0.731), f32)
    # Below float32 precision two rewards are the same (documented limitation).
    assert reward_bits(1.0, f32) == reward_bits(1.0 + 1e-12, f32)
    f64 = reward_dtype_of(np.float64)
    assert reward_bits(1.0, f64) != reward_bits(1.0 + 1e-12, f64)
    with pytest.raises(ValueError):
        reward_dtype_of(np.complex128)
