"""Hand of Straights - LeetCode 846 - Reference Solution"""

from collections import Counter

def is_n_straight_hand(hand: list[int], group_size: int) -> bool:
    if len(hand) % group_size != 0:
        return False

    count = Counter(hand)

    for card in sorted(count.keys()):
        if count[card] > 0:
            start_count = count[card]
            for i in range(group_size):
                if count[card + i] < start_count:
                    return False
                count[card + i] -= start_count

    return True
