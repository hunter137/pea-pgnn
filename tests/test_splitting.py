import numpy as np
import pytest

from pea_pgnn import GroupedSplit, grouped_train_validation_test_split


def test_grouped_split_is_reproducible_complete_and_leakage_free() -> None:
    groups = np.repeat([f"mix-{index:02d}" for index in range(20)], 4)

    first = grouped_train_validation_test_split(groups, seed=7)
    second = grouped_train_validation_test_split(groups, seed=7)

    assert isinstance(first, GroupedSplit)
    np.testing.assert_array_equal(first.train_indices, second.train_indices)
    np.testing.assert_array_equal(first.validation_indices, second.validation_indices)
    np.testing.assert_array_equal(first.test_indices, second.test_indices)

    all_indices = np.concatenate(
        [first.train_indices, first.validation_indices, first.test_indices]
    )
    np.testing.assert_array_equal(np.sort(all_indices), np.arange(len(groups)))

    train_groups = set(groups[first.train_indices])
    validation_groups = set(groups[first.validation_indices])
    test_groups = set(groups[first.test_indices])
    assert train_groups.isdisjoint(validation_groups)
    assert train_groups.isdisjoint(test_groups)
    assert validation_groups.isdisjoint(test_groups)
    assert len(train_groups) == 12
    assert len(validation_groups) == 4
    assert len(test_groups) == 4


@pytest.mark.parametrize(
    ("validation_fraction", "test_fraction"),
    [(0.0, 0.2), (0.2, 0.0), (0.8, 0.2), (1.0, 0.1)],
)
def test_grouped_split_rejects_invalid_fractions(
    validation_fraction: float, test_fraction: float
) -> None:
    with pytest.raises(ValueError):
        grouped_train_validation_test_split(
            ["a", "b", "c", "d"],
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )


def test_grouped_split_rejects_bad_groups_and_seed() -> None:
    with pytest.raises(ValueError, match="three distinct"):
        grouped_train_validation_test_split(["a", "a", "b", "b"])
    with pytest.raises(ValueError, match="missing"):
        grouped_train_validation_test_split(["a", "b", None, "c"])
    with pytest.raises(ValueError, match="one-dimensional"):
        grouped_train_validation_test_split([["a"], ["b"], ["c"]])
    with pytest.raises(ValueError, match="seed"):
        grouped_train_validation_test_split(["a", "b", "c"], seed=-1)
