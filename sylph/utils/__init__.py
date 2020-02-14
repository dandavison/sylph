from collections import Counter
from typing import List
from typing import TypeVar

T = TypeVar("T")


def get_group_modes(values: List[T], group_sizes: List[int]) -> List[T]:
    assert sum(group_sizes) == len(values)
    offset = 0
    modes = []
    for n in group_sizes:
        group = values[offset: (offset + n)]
        mode, cnt = Counter(group).most_common()[0]
        modes.append(mode)
        offset += n
    return modes
