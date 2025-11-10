"""Course Schedule II - LeetCode 210 - Reference Solution"""

def find_order(num_courses: int, prerequisites: list[list[int]]) -> list[int]:
    # Build adjacency list
    prereq_map = {i: [] for i in range(num_courses)}
    for course, prereq in prerequisites:
        prereq_map[course].append(prereq)

    visited = set()
    cycle = set()
    order = []

    def dfs(course: int) -> bool:
        if course in cycle:
            return False  # Cycle detected

        if course in visited:
            return True

        cycle.add(course)
        for prereq in prereq_map[course]:
            if not dfs(prereq):
                return False

        cycle.remove(course)
        visited.add(course)
        order.append(course)
        return True

    for course in range(num_courses):
        if not dfs(course):
            return []

    return order
