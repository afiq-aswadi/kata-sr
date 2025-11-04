//! Dependency graph management for kata prerequisites.
//!
//! This module provides a directed graph structure for managing prerequisite
//! relationships between katas. A kata can depend on other katas being completed
//! successfully a specified number of times before it becomes available.
//!
//! # Examples
//!
//! ```
//! # use kata_sr::core::dependencies::DependencyGraph;
//! # use std::collections::HashMap;
//! let mut graph = DependencyGraph::new();
//!
//! // Kata 2 requires Kata 1 to be completed once
//! graph.add_dependency(2, 1, 1);
//!
//! // Check if kata 2 is unlocked
//! let mut counts = HashMap::new();
//! assert_eq!(graph.is_unlocked(2, &counts), false);
//!
//! // After completing kata 1, kata 2 becomes unlocked
//! counts.insert(1, 1);
//! assert_eq!(graph.is_unlocked(2, &counts), true);
//! ```

use std::collections::{HashMap, HashSet, VecDeque};

/// Manages prerequisite dependencies between katas.
///
/// Stores directed edges where each kata can depend on multiple other katas,
/// each with a required success count. Provides methods to check unlock status
/// and identify blocking dependencies.
#[derive(Debug, Clone)]
pub struct DependencyGraph {
    dependencies: HashMap<i64, Vec<(i64, i64)>>,
}

impl DependencyGraph {
    /// Creates a new empty dependency graph.
    ///
    /// # Examples
    ///
    /// ```
    /// # use kata_sr::core::dependencies::DependencyGraph;
    /// let graph = DependencyGraph::new();
    /// ```
    pub fn new() -> Self {
        Self {
            dependencies: HashMap::new(),
        }
    }

    /// Adds a dependency relationship to the graph.
    ///
    /// Specifies that `kata_id` requires `depends_on` to be completed
    /// successfully `required_count` times before it becomes available.
    ///
    /// # Arguments
    ///
    /// * `kata_id` - The kata that has the dependency
    /// * `depends_on` - The prerequisite kata ID
    /// * `required_count` - Number of successful completions required
    ///
    /// # Examples
    ///
    /// ```
    /// # use kata_sr::core::dependencies::DependencyGraph;
    /// let mut graph = DependencyGraph::new();
    ///
    /// // Kata 3 requires kata 1 to be completed 2 times
    /// graph.add_dependency(3, 1, 2);
    ///
    /// // Kata 3 also requires kata 2 to be completed once
    /// graph.add_dependency(3, 2, 1);
    /// ```
    pub fn add_dependency(&mut self, kata_id: i64, depends_on: i64, required_count: i64) {
        self.dependencies
            .entry(kata_id)
            .or_default()
            .push((depends_on, required_count));
    }

    /// Checks if a kata is unlocked based on current success counts.
    ///
    /// A kata is unlocked if all of its dependencies have been satisfied
    /// (each prerequisite has been completed the required number of times).
    /// A kata with no dependencies is always unlocked.
    ///
    /// # Arguments
    ///
    /// * `kata_id` - The kata to check
    /// * `success_counts` - Map of kata ID to number of successful completions
    ///
    /// # Examples
    ///
    /// ```
    /// # use kata_sr::core::dependencies::DependencyGraph;
    /// # use std::collections::HashMap;
    /// let mut graph = DependencyGraph::new();
    /// graph.add_dependency(2, 1, 2);
    ///
    /// let mut counts = HashMap::new();
    /// counts.insert(1, 1);
    /// assert_eq!(graph.is_unlocked(2, &counts), false);
    ///
    /// counts.insert(1, 2);
    /// assert_eq!(graph.is_unlocked(2, &counts), true);
    /// ```
    pub fn is_unlocked(&self, kata_id: i64, success_counts: &HashMap<i64, i64>) -> bool {
        if let Some(deps) = self.dependencies.get(&kata_id) {
            for (dep_id, required) in deps {
                let count = success_counts.get(dep_id).unwrap_or(&0);
                if count < required {
                    return false;
                }
            }
        }
        true
    }

    /// Returns all unsatisfied dependencies for a kata.
    ///
    /// Each blocking dependency is returned as a tuple of:
    /// (prerequisite_id, required_count, current_count)
    ///
    /// # Arguments
    ///
    /// * `kata_id` - The kata to check
    /// * `success_counts` - Map of kata ID to number of successful completions
    ///
    /// # Examples
    ///
    /// ```
    /// # use kata_sr::core::dependencies::DependencyGraph;
    /// # use std::collections::HashMap;
    /// let mut graph = DependencyGraph::new();
    /// graph.add_dependency(3, 1, 2);
    /// graph.add_dependency(3, 2, 1);
    ///
    /// let mut counts = HashMap::new();
    /// counts.insert(1, 1);
    /// // Kata 2 has 0 successes (not in map)
    ///
    /// let blocking = graph.get_blocking_dependencies(3, &counts);
    /// assert_eq!(blocking.len(), 2);
    /// assert!(blocking.contains(&(1, 2, 1)));
    /// assert!(blocking.contains(&(2, 1, 0)));
    /// ```
    pub fn get_blocking_dependencies(
        &self,
        kata_id: i64,
        success_counts: &HashMap<i64, i64>,
    ) -> Vec<(i64, i64, i64)> {
        let mut blocking = Vec::new();

        if let Some(deps) = self.dependencies.get(&kata_id) {
            for (dep_id, required) in deps {
                let current = *success_counts.get(dep_id).unwrap_or(&0);
                if current < *required {
                    blocking.push((*dep_id, *required, current));
                }
            }
        }

        blocking
    }

    /// Performs topological sort on the dependency graph.
    ///
    /// Returns katas in an order where all dependencies come before dependents.
    /// Useful for suggesting a learning path. Returns an error if a cycle is detected.
    ///
    /// Uses Kahn's algorithm for topological sorting with cycle detection.
    ///
    /// # Returns
    ///
    /// - `Ok(Vec<i64>)` - Sorted list of kata IDs
    /// - `Err(String)` - Error message if cycle detected
    ///
    /// # Examples
    ///
    /// ```
    /// # use kata_sr::core::dependencies::DependencyGraph;
    /// let mut graph = DependencyGraph::new();
    /// graph.add_dependency(2, 1, 1);
    /// graph.add_dependency(3, 2, 1);
    ///
    /// let sorted = graph.topological_sort().unwrap();
    /// // 1 must come before 2, and 2 must come before 3
    /// let pos_1 = sorted.iter().position(|&x| x == 1).unwrap();
    /// let pos_2 = sorted.iter().position(|&x| x == 2).unwrap();
    /// let pos_3 = sorted.iter().position(|&x| x == 3).unwrap();
    /// assert!(pos_1 < pos_2);
    /// assert!(pos_2 < pos_3);
    /// ```
    pub fn topological_sort(&self) -> Result<Vec<i64>, String> {
        let mut in_degree: HashMap<i64, usize> = HashMap::new();
        let mut all_nodes: HashSet<i64> = HashSet::new();
        let mut adjacency: HashMap<i64, Vec<i64>> = HashMap::new();

        // build adjacency list and compute in-degrees
        for (&kata_id, deps) in &self.dependencies {
            all_nodes.insert(kata_id);
            in_degree.entry(kata_id).or_insert(0);

            for (dep_id, _) in deps {
                all_nodes.insert(*dep_id);
                adjacency.entry(*dep_id).or_default().push(kata_id);
                *in_degree.entry(kata_id).or_insert(0) += 1;
            }
        }

        // initialize in_degree for nodes with no dependencies
        for &node in &all_nodes {
            in_degree.entry(node).or_insert(0);
        }

        // start with nodes that have no dependencies
        let mut queue: VecDeque<i64> = in_degree
            .iter()
            .filter(|(_, &degree)| degree == 0)
            .map(|(&node, _)| node)
            .collect();

        let mut sorted = Vec::new();

        while let Some(node) = queue.pop_front() {
            sorted.push(node);

            if let Some(neighbors) = adjacency.get(&node) {
                for &neighbor in neighbors {
                    if let Some(degree) = in_degree.get_mut(&neighbor) {
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
        }

        if sorted.len() != all_nodes.len() {
            return Err("Cycle detected in dependency graph".to_string());
        }

        Ok(sorted)
    }

    /// Returns all kata IDs that have dependencies in this graph.
    pub fn get_all_kata_ids(&self) -> HashSet<i64> {
        let mut ids = HashSet::new();
        for (&kata_id, deps) in &self.dependencies {
            ids.insert(kata_id);
            for (dep_id, _) in deps {
                ids.insert(*dep_id);
            }
        }
        ids
    }
}

impl Default for DependencyGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_unlock() {
        let mut graph = DependencyGraph::new();
        graph.add_dependency(2, 1, 1);

        let mut counts = HashMap::new();
        assert!(!graph.is_unlocked(2, &counts));

        counts.insert(1, 1);
        assert!(graph.is_unlocked(2, &counts));
    }

    #[test]
    fn test_multiple_dependencies() {
        let mut graph = DependencyGraph::new();
        graph.add_dependency(3, 1, 1);
        graph.add_dependency(3, 2, 1);

        let mut counts = HashMap::new();
        assert!(!graph.is_unlocked(3, &counts));

        counts.insert(1, 1);
        assert!(!graph.is_unlocked(3, &counts));

        counts.insert(2, 1);
        assert!(graph.is_unlocked(3, &counts));
    }

    #[test]
    fn test_required_count() {
        let mut graph = DependencyGraph::new();
        graph.add_dependency(2, 1, 3);

        let mut counts = HashMap::new();
        counts.insert(1, 1);
        assert!(!graph.is_unlocked(2, &counts));

        counts.insert(1, 2);
        assert!(!graph.is_unlocked(2, &counts));

        counts.insert(1, 3);
        assert!(graph.is_unlocked(2, &counts));
    }

    #[test]
    fn test_no_dependencies_always_unlocked() {
        let graph = DependencyGraph::new();
        let counts = HashMap::new();
        assert!(graph.is_unlocked(1, &counts));
    }

    #[test]
    fn test_blocking_dependencies() {
        let mut graph = DependencyGraph::new();
        graph.add_dependency(3, 1, 2);
        graph.add_dependency(3, 2, 1);

        let mut counts = HashMap::new();
        counts.insert(1, 1);

        let blocking = graph.get_blocking_dependencies(3, &counts);
        assert_eq!(blocking.len(), 2);
        assert!(blocking.contains(&(1, 2, 1)));
        assert!(blocking.contains(&(2, 1, 0)));
    }

    #[test]
    fn test_no_blocking_when_unlocked() {
        let mut graph = DependencyGraph::new();
        graph.add_dependency(2, 1, 1);

        let mut counts = HashMap::new();
        counts.insert(1, 1);

        let blocking = graph.get_blocking_dependencies(2, &counts);
        assert_eq!(blocking.len(), 0);
    }

    #[test]
    fn test_topological_sort_simple() {
        let mut graph = DependencyGraph::new();
        graph.add_dependency(2, 1, 1);

        let sorted = graph.topological_sort().unwrap();
        let pos_1 = sorted.iter().position(|&x| x == 1).unwrap();
        let pos_2 = sorted.iter().position(|&x| x == 2).unwrap();
        assert!(pos_1 < pos_2);
    }

    #[test]
    fn test_topological_sort_chain() {
        let mut graph = DependencyGraph::new();
        graph.add_dependency(2, 1, 1);
        graph.add_dependency(3, 2, 1);
        graph.add_dependency(4, 3, 1);

        let sorted = graph.topological_sort().unwrap();
        let pos_1 = sorted.iter().position(|&x| x == 1).unwrap();
        let pos_2 = sorted.iter().position(|&x| x == 2).unwrap();
        let pos_3 = sorted.iter().position(|&x| x == 3).unwrap();
        let pos_4 = sorted.iter().position(|&x| x == 4).unwrap();

        assert!(pos_1 < pos_2);
        assert!(pos_2 < pos_3);
        assert!(pos_3 < pos_4);
    }

    #[test]
    fn test_topological_sort_detects_cycle() {
        let mut graph = DependencyGraph::new();
        graph.add_dependency(1, 2, 1);
        graph.add_dependency(2, 1, 1);

        let result = graph.topological_sort();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Cycle"));
    }

    #[test]
    fn test_topological_sort_complex() {
        let mut graph = DependencyGraph::new();
        //     1
        //    / \
        //   2   3
        //    \ /
        //     4
        graph.add_dependency(2, 1, 1);
        graph.add_dependency(3, 1, 1);
        graph.add_dependency(4, 2, 1);
        graph.add_dependency(4, 3, 1);

        let sorted = graph.topological_sort().unwrap();
        let pos_1 = sorted.iter().position(|&x| x == 1).unwrap();
        let pos_2 = sorted.iter().position(|&x| x == 2).unwrap();
        let pos_3 = sorted.iter().position(|&x| x == 3).unwrap();
        let pos_4 = sorted.iter().position(|&x| x == 4).unwrap();

        assert!(pos_1 < pos_2);
        assert!(pos_1 < pos_3);
        assert!(pos_2 < pos_4);
        assert!(pos_3 < pos_4);
    }

    #[test]
    fn test_get_all_kata_ids() {
        let mut graph = DependencyGraph::new();
        graph.add_dependency(2, 1, 1);
        graph.add_dependency(3, 1, 1);

        let ids = graph.get_all_kata_ids();
        assert_eq!(ids.len(), 3);
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert!(ids.contains(&3));
    }
}
