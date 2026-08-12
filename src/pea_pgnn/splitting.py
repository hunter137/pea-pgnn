"""Leakage-resistant data splitting for repeated engineering conditions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.model_selection import GroupShuffleSplit


@dataclass(frozen=True)
class GroupedSplit:
    """Row indices for mutually group-disjoint train, validation, and test sets."""

    train_indices: NDArray[np.int64]
    validation_indices: NDArray[np.int64]
    test_indices: NDArray[np.int64]


def _encode_groups(groups: ArrayLike) -> NDArray[np.int64]:
    group_array = np.asarray(groups, dtype=object)
    if group_array.ndim != 1 or group_array.size == 0:
        raise ValueError("groups must be a non-empty one-dimensional array")

    codes = np.empty(group_array.size, dtype=np.int64)
    code_by_group: dict[Any, int] = {}
    for index, value in enumerate(group_array.tolist()):
        if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
            raise ValueError("groups cannot contain missing values")
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError("each group identifier must be a hashable scalar") from exc
        if value not in code_by_group:
            code_by_group[value] = len(code_by_group)
        codes[index] = code_by_group[value]

    if len(code_by_group) < 3:
        raise ValueError("at least three distinct groups are required")
    return codes


def grouped_train_validation_test_split(
    groups: ArrayLike,
    *,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> GroupedSplit:
    """Split rows while keeping every physical condition in exactly one set.

    Parameters
    ----------
    groups:
        One group identifier per row, such as a concrete ``mix_id``. Repeated
        ages from the same condition must carry the same identifier.
    validation_fraction, test_fraction:
        Requested fractions of all rows. Whole groups are assigned, so the
        realized row fractions can differ when group sizes are unequal.
    seed:
        Non-negative random seed used for reproducible group assignment.

    Returns
    -------
    GroupedSplit
        Integer row indices. No group identifier occurs in more than one set.
    """

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be less than 1")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    group_codes = _encode_groups(groups)
    indices = np.arange(group_codes.size, dtype=np.int64)

    outer = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    development_relative, test_relative = next(
        outer.split(indices, groups=group_codes)
    )
    development_indices = indices[development_relative]
    test_indices = indices[test_relative]

    relative_validation_fraction = validation_fraction / (1.0 - test_fraction)
    inner_seed = (seed + 1) % (2**32 - 1)
    inner = GroupShuffleSplit(
        n_splits=1,
        test_size=relative_validation_fraction,
        random_state=inner_seed,
    )
    train_relative, validation_relative = next(
        inner.split(
            development_indices,
            groups=group_codes[development_indices],
        )
    )
    train_indices = development_indices[train_relative]
    validation_indices = development_indices[validation_relative]

    split_group_sets = (
        set(group_codes[train_indices]),
        set(group_codes[validation_indices]),
        set(group_codes[test_indices]),
    )
    if any(
        split_group_sets[left] & split_group_sets[right]
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise RuntimeError("group leakage occurred while constructing the split")

    return GroupedSplit(
        train_indices=np.sort(train_indices),
        validation_indices=np.sort(validation_indices),
        test_indices=np.sort(test_indices),
    )
