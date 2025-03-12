### **Easy Problems**  

1. **Fibonacci Numbers**  
   - Compute the **nth Fibonacci number**, where the sequence follows:  
     ```
     F(n) = F(n-1) + F(n-2), 
     F(0) = 0, 
     F(1) = 1
     ```
   - Use **recursion** or **dynamic programming** for optimal solutions.

2. **Climbing Stairs**  
   - You can take **1 or 2 steps** at a time. Find the number of distinct ways to reach the **nth** step.
   - Example:  
     - **Input:** `n = 3`  
     - **Output:** `3` (Ways: `[1,1,1], [1,2], [2,1]`)

3. **Coin Change (Ways to Make Change)**  
   - Given an array of **coin denominations** and an amount, find the number of ways to make the amount.
   - Example:  
     - **Input:** `coins = [1,2,5]`, `amount = 5`  
     - **Output:** `4`  
     - Ways: `{5}, {2,2,1}, {2,1,1,1}, {1,1,1,1,1}`

4. **House Robber**  
   - Given an array of **house values**, find the **maximum sum** you can rob **without robbing adjacent houses**.
   - Example:  
     - **Input:** `houses = [2,7,9,3,1]`  
     - **Output:** `12` (`2 + 9 + 1`)

5. **Jump Game**  
   - Given an array where each element represents **maximum jump length**, determine if you can reach the **last index**.
   - Example:  
     - **Input:** `nums = [2,3,1,1,4]`  
     - **Output:** `true` (Jump `2 → 3 → 4`)

6. **Min Cost Climbing Stairs**  
   - Given an array where `cost[i]` represents the cost to step on stair `i`, find the **minimum cost** to reach the top.
   - Example:  
     - **Input:** `cost = [10,15,20]`  
     - **Output:** `15` (Choose `15` → skip `20`)

7. **Unique Paths**  
   - Find the number of ways to move from the **top-left** to **bottom-right** of an `m × n` grid **only moving right or down**.
   - Example:  
     - **Input:** `m = 3, n = 2`  
     - **Output:** `3` (`→ ↓ ↓`, `↓ → ↓`, `↓ ↓ →`)

8. **Longest Common Subsequence (LCS)**  
   - Given two strings, find the **longest subsequence** common to both.
   - Example:  
     - **Input:** `s1 = "abcde"`, `s2 = "ace"`  
     - **Output:** `3` (`ace`)

9. **Edit Distance (Levenshtein Distance)**  
   - Given two strings, find the **minimum number of insertions, deletions, and substitutions** to transform one string into another.
   - Example:  
     - **Input:** `"horse"`, `"ros"`  
     - **Output:** `3` (Operations: `horse → rorse → rose → ros`)

10. **0/1 Knapsack Problem**  
    - Given items with **weights and values**, maximize the total value in a **knapsack** with a given **weight limit**.
    - Example:  
      - **Input:** `weights = [1,3,4,5]`, `values = [1,4,5,7]`, `W = 7`  
      - **Output:** `9` (`items[3,4]`)

---

### **Medium Problems**  

11. **Partition Equal Subset Sum**  
    - Determine if an array can be partitioned into **two subsets of equal sum**.

12. **Rod Cutting**  
    - Given a rod of length `n` and a list of **prices** for each length, determine the **maximum obtainable price**.

13. **Longest Increasing Subsequence (LIS)**  
    - Find the **length** of the longest **increasing subsequence** in an array.

14. **Palindromic Substrings**  
    - Count the number of **palindromic substrings** in a string.

15. **Longest Palindromic Subsequence**  
    - Find the **longest subsequence** in a string that is a **palindrome**.

16. **Subset Sum Problem**  
    - Determine if there exists a subset with a sum equal to a **given value**.

17. **Egg Dropping Problem**  
    - Given `k` eggs and `n` floors, determine the **minimum number of drops** needed to find the critical floor.

18. **Wildcard Matching**  
    - Check if a string matches a **pattern** with `*` (matches any sequence) and `?` (matches any character).

19. **Interleaving Strings**  
    - Check if a string is formed by **interleaving** two other strings.

20. **Burst Balloons**  
    - Find the **maximum coins** obtained by bursting balloons in an optimal order.

---

### **Hard Problems**  

21. **Word Break II**  
    - Given a dictionary, return **all possible ways** to segment a string into valid words.

22. **Maximal Rectangle**  
    - Find the **largest rectangle** containing only `1s` in a binary matrix.

23. **Russian Doll Envelopes**  
    - Find the maximum number of **nested** envelopes.

24. **Minimum Window Substring**  
    - Find the **smallest substring** of `s` that contains all characters of `t`.

25. **Regular Expression Matching**  
    - Implement regex matching with `.` and `*`.

26. **Palindrome Partitioning II**  
    - Find the **minimum cuts** to partition a string into palindromes.

27. **Number of Ways to Arrange Buildings**  
    - Count valid arrangements of buildings on **both sides of a street**.

28. **Scramble String**  
    - Determine if one string is a **scrambled version** of another.

29. **Travelling Salesman Problem (TSP)**  
    - Find the **shortest path** to visit all cities once and return to the start.

30. **Optimal Binary Search Tree**  
    - Construct a **BST** that minimizes search costs given **access frequency**.

---