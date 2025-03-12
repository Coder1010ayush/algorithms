# Dynamic Programming (DP) Guide

## Introduction
Dynamic Programming (DP) is an optimization technique used to solve complex problems by breaking them down into smaller overlapping subproblems. It is typically used when a problem has the following characteristics:
- **Overlapping Subproblems**: The problem can be broken down into smaller subproblems that are reused multiple times.
- **Optimal Substructure**: The optimal solution of the problem can be constructed from optimal solutions of its subproblems.

DP problems are usually solved using one of the two approaches:
1. **Top-Down Approach (Memoization)**: Solve the problem recursively and store results to avoid redundant computations.
2. **Bottom-Up Approach (Tabulation)**: Solve subproblems iteratively and store results in a table to build up to the final solution.

---

## Categories of DP Problems
### 1. **Basic DP Problems**
- **Fibonacci Sequence** (Recursive + Memoization, Iterative)
- **Factorial of a Number**
- **Binomial Coefficient Calculation**

### 2. **1D DP Problems**
- **Climbing Stairs** (Ways to climb N stairs with 1 or 2 steps at a time)
- **House Robber** (Maximize sum by robbing non-adjacent houses)
- **Maximum Subarray (Kadane's Algorithm)** (Find the maximum sum subarray)
- **Coin Change** (Minimum number of coins to make amount N)

### 3. **2D DP Problems**
- **Longest Common Subsequence (LCS)** (Find longest subsequence common to two strings)
- **Longest Palindromic Subsequence**
- **Edit Distance (Levenshtein Distance)** (Minimum operations to convert one string to another)
- **Knapsack Problem (0/1 Knapsack)**

### 4. **Grid-based DP Problems**
- **Unique Paths** (Count paths in an MxN grid with right/down movements)
- **Minimum Path Sum** (Find the minimum cost path in a grid)

### 5. **DP on Trees**
- **Binary Tree Maximum Path Sum**
- **Diameter of a Binary Tree**
- **Tree DP (Finding the largest independent set, etc.)**

### 6. **DP on Graphs**
- **Floyd-Warshall Algorithm** (All-pairs shortest paths)
- **Bellman-Ford Algorithm** (Single-source shortest path for graphs with negative weights)

### 7. **Advanced DP Problems**
- **Matrix Chain Multiplication**
- **Egg Dropping Puzzle**
- **Partition DP** (Palindrome Partitioning, Burst Balloons, etc.)

---

## Example Problems and Solutions
### 1. Fibonacci Numbers (Top-Down and Bottom-Up)
#### **Recursive + Memoization Approach**
```python
from functools import lru_cache

def fibonacci(n):
    @lru_cache(None)
    def helper(x):
        if x <= 1:
            return x
        return helper(x-1) + helper(x-2)
    return helper(n)

print(fibonacci(10))  
```

#### **Bottom-Up (Tabulation) Approach**
```python
def fibonacci(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

print(fibonacci(10))  
```

---

## Tips for Solving DP Problems
1. **Identify Overlapping Subproblems**: If a problem can be broken into smaller subproblems that are reused, DP might be applicable.
2. **Define the State**: Determine the variables that represent the problem's state.
3. **Formulate Recurrence Relation**: Express the problem in terms of its subproblems.
4. **Choose Memoization or Tabulation**: Decide whether to solve recursively with caching or iteratively using a table.
5. **Optimize Space Complexity**: If only a few previous states are needed, reduce the DP table size.

---

## Resources for Further Learning
- [Introduction to DP - GeeksforGeeks](https://www.geeksforgeeks.org/dynamic-programming/)
- [MIT OpenCourseWare - DP Lectures](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-006-introduction-to-algorithms-fall-2011/)
- [LeetCode DP Problems](https://leetcode.com/tag/dynamic-programming/)

---

This guide provides a structured approach to understanding and solving dynamic programming problems. Happy coding!

