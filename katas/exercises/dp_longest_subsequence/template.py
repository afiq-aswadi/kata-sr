"""Longest Common Subsequence using dynamic programming."""


def lcs_length(s1: str, s2: str) -> int:
    """Find length of longest common subsequence.

    Use dynamic programming with a 2D table.
    LCS of "ABCD" and "ACDF" is "ACD" (length 3).

    Args:
        s1: first string
        s2: second string

    Returns:
        length of longest common subsequence
    """
    # TODO: Implement using 2D DP table
    # Hint: dp[i][j] = LCS length of s1[:i] and s2[:j]
    # If s1[i-1] == s2[j-1]: dp[i][j] = dp[i-1][j-1] + 1
    # Else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    # BLANK_START
    raise NotImplementedError
    # BLANK_END
