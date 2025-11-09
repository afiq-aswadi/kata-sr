"""Edit distance (Levenshtein distance) using dynamic programming."""


def edit_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein distance between two strings.

    Minimum number of single-character edits (insertions, deletions,
    or substitutions) needed to transform s1 into s2.

    Args:
        s1: first string
        s2: second string

    Returns:
        minimum edit distance
    """
    # TODO: Implement using 2D DP table
    # Hint: dp[i][j] = edit distance between s1[:i] and s2[:j]
    # If s1[i-1] == s2[j-1]: dp[i][j] = dp[i-1][j-1]
    # Else: dp[i][j] = 1 + min(
    #   dp[i-1][j],    # delete from s1
    #   dp[i][j-1],    # insert into s1
    #   dp[i-1][j-1]   # substitute
    # )
    # BLANK_START
    raise NotImplementedError
    # BLANK_END
