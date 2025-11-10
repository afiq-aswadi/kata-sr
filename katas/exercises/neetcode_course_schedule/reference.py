"""Course Schedule - LeetCode 207 - Reference Solution"""

def can_finish(num_courses: int, prerequisites: list[list[int]]) -> bool:
    # Build adjacency list
    prereq_map = {i: [] for i in range(num_courses)}
    for course, prereq in prerequisites:
        prereq_map[course].append(prereq)

    visited = set()

    def dfs(course: int) -> bool:
        if course in visited:
            return False  # Cycle detected

        if not prereq_map[course]:
            return True  # No prerequisites

        visited.add(course)
        for prereq in prereq_map[course]:
            if not dfs(prereq):
                return False

        visited.remove(course)
        prereq_map[course] = []  # Mark as processed
        return True

    for course in range(num_courses):
        if not dfs(course):
            return False

    return True
