"""Best Time to Buy and Sell Stock with Cooldown - LeetCode 309 - Reference Solution"""

def max_profit(prices: list[int]) -> int:
    if not prices:
        return 0

    # State: sold (just sold), hold (holding stock), rest (cooldown/no stock)
    sold = 0
    hold = -prices[0]
    rest = 0

    for i in range(1, len(prices)):
        prev_sold = sold
        sold = hold + prices[i]
        hold = max(hold, rest - prices[i])
        rest = max(rest, prev_sold)

    return max(sold, rest)
