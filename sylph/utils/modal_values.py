from collections import Counter
from collections import defaultdict
from typing import List
from typing import Dict
from typing import TypeVar

K = TypeVar("K")
V = TypeVar("V")


def get_modal_values(group_ids: List[K], values: List[V]) -> Dict[K, V]:
    """
    Return dict mapping group label to modal group value.

    Example:

    >>> group_ids = ["A", "A", "A", "B", "B", "B"]
    >>> values = [1, 1, 2, 2, 2, 1]
    >>> get_modal_values(group_ids, values)
    {'A': 1, 'B': 2}
    """
    assert len(group_ids) == len(values)
    group2counts: dict = defaultdict(Counter)
    for k, v in zip(group_ids, values):
        group2counts[k][v] += 1
    return {k: counts.most_common()[0][0] for k, counts in group2counts.items()}
