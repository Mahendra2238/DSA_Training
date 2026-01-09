# 1. LeetCode 121 – Best Time to Buy and Sell Stock
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        maxp = 0  # stores maximum profit so far
        minbuy = prices[0]  # minimum price to buy till current day
        for i in range(1, len(prices)):
            minbuy = min(minbuy, prices[i])  # update minimum buying price
            maxp = max(maxp, prices[i] - minbuy)  # update max profit if selling today
        return maxp  # final maximum profit
# Time Complexity: O(n)
# Space Complexity: O(1)
# Algo: Track minimum price so far, compute profit for each day, keep the maximum

# LeetCode 494 – Target Sum (Clear & Simple DP)
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        dp = {0: 1}  # dp[sum] = number of ways to form this sum
        for num in nums:
            nextDP = {}  # stores sums for current number
            for s, cnt in dp.items():  # s = previous sum, cnt = ways to get that sum
                nextDP[s + num] = nextDP.get(s + num, 0) + cnt  # add '+' sign to num
                nextDP[s - num] = nextDP.get(s - num, 0) + cnt  # add '-' sign to num
            dp = nextDP  # move to next iteration
        return dp.get(target, 0)  # number of ways to reach target
# Time Complexity: O(n * range)
# Space Complexity: O(range)
# Algo: Use hashmap DP, for each number create new sums by +num and -num, count ways

# 0/1 Knapsack Problem
# Question:
# Given weights wt[], values val[] of n items and a knapsack capacity W,
# find the maximum total value that can be put into the knapsack.
# Each item can be taken at most once.

def knapsack(W, wt, val, n):
    dp = [0] * (W + 1)  # dp[w] = maximum value with capacity w
    for i in range(n):  # iterate over each item
        for w in range(W, wt[i] - 1, -1):  # traverse capacity backwards
            dp[w] = max(dp[w], val[i] + dp[w - wt[i]])  # take or skip current item
    return dp[W]  # maximum value achievable

# Example Input:
# n = 3
# wt = [4, 5, 1]
# val = [1, 2, 3]
# W = 4
# Output:
# 3
# Time Complexity: O(n * W)
# Space Complexity: O(W)
# Algo: Use 1D DP, iterate items, update capacities backward to ensure 0/1 constraint

# Fractional Knapsack Problem
# Question:
# Given weights wt[], values val[] of n items and knapsack capacity W,
# you can take fractions of items. Find the maximum total value.

def fractionalKnapsack(W, wt, val, n):
    items = []  # (value/weight, value, weight)
    for i in range(n):
        items.append((val[i] / wt[i], val[i], wt[i]))  # compute ratio
    items.sort(reverse=True)  # sort by highest value/weight
    maxValue = 0.0  # total value
    for ratio, value, weight in items:
        if W == 0: break  # knapsack full
        if weight <= W:
            maxValue += value  # take whole item
            W -= weight
        else:
            maxValue += ratio * W  # take fraction
            W = 0
    return maxValue

# Example Input:
# n = 3
# wt = [4, 5, 1]
# val = [1, 2, 3]
# W = 4
# Output:
# 3.0
# Time Complexity: O(n log n)
# Space Complexity: O(n)
# Algo: Greedy – take items in decreasing order of value/weight ratio

# Unbounded Knapsack Problem
# Meaning:
# You are given weights wt[], values val[] and capacity W.
# Each item can be taken unlimited number of times.

def unboundedKnapsack(W, wt, val, n):
    dp = [0] * (W + 1)  # dp[w] = maximum value for capacity w
    for w in range(W + 1):  # iterate over all capacities
        for i in range(n):  # try every item
            if wt[i] <= w:
                dp[w] = max(dp[w], val[i] + dp[w - wt[i]])  # take item again
    return dp[W]

# Example Input:
# n = 2
# wt = [2, 3]
# val = [4, 5]
# W = 7
# Output:
# 12
# Time Complexity: O(n * W)
# Space Complexity: O(W)
# Algo: Use 1D DP, for each capacity try all items (items can be reused)

# LeetCode 983 – Minimum Cost For Tickets
class Solution:
    def mincostTickets(self, days: List[int], costs: List[int]) -> int:
        st = set(days)  # fast lookup for travel days
        last_day = days[-1]  # last day of travel
        t = [0] * (last_day + 1)  # t[i] = min cost to cover up to day i
        for i in range(1, last_day + 1):
            if i not in st:  # if no travel on day i
                t[i] = t[i - 1]  # cost remains same as previous day
                continue
            day1_pass = costs[0] + t[max(i - 1, 0)]  # 1-day pass
            day7_pass = costs[1] + t[max(i - 7, 0)]  # 7-day pass
            day30_pass = costs[2] + t[max(i - 30, 0)]  # 30-day pass
            t[i] = min(day1_pass, day7_pass, day30_pass)  # choose minimum
        return t[last_day]  # minimum cost to cover all travel days
# Time Complexity: O(last_day)
# Space Complexity: O(last_day)
# Algo: DP where each day chooses the cheapest ticket covering that day

# LeetCode 322 – Coin Change
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        max_value = amount + 1              # sentinel value for unreachable states
        dp = [max_value] * (amount + 1)     # dp[i] = minimum coins needed for amount i
        dp[0] = 0                           # base case: 0 coins to make amount 0
        for i in range(1, amount + 1):      # build DP bottom-up
            for coin in coins:              # try using each coin
                if coin <= i:
                    dp[i] = min(dp[i], 1 + dp[i - coin])  # take current coin
        return -1 if dp[amount] == max_value else dp[amount]  # check feasibility
# Time Complexity: O(amount * len(coins))
# Space Complexity: O(amount)
# Algo: Unbounded knapsack DP, each amount picks the best coin option

# LeetCode 718 – Maximum Length of Repeated Subarray
class Solution:
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        m = len(nums1)  # length of first array
        n = len(nums2)  # length of second array
        dp = [[0] * (n + 1) for _ in range(m + 1)]  # dp[i][j] = length of common subarray ending at i-1, j-1
        maxl = 0  # store maximum length found
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if nums1[i - 1] == nums2[j - 1]:  # if elements match
                    dp[i][j] = 1 + dp[i - 1][j - 1]  # extend previous subarray
                    maxl = max(maxl, dp[i][j])  # update maximum
        return maxl  # longest repeated subarray length
# Time Complexity: O(m * n)
# Space Complexity: O(m * n)
# Algo: DP similar to LCS but reset to 0 on mismatch (subarray must be contiguous)

# LeetCode 72 – Edit Distance
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m = len(word1)  # length of word1
        n = len(word2)  # length of word2
        dp = [[0] * (n + 1) for _ in range(m + 1)]  # dp[i][j] = min ops to convert word1[0:i] → word2[0:j]
        for i in range(m + 1):
            dp[i][0] = i  # delete all characters from word1
        for j in range(n + 1):
            dp[0][j] = j  # insert all characters to word1
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:  # characters match
                    dp[i][j] = dp[i - 1][j - 1]  # no operation needed
                else:
                    replace = dp[i - 1][j - 1]  # replace character
                    delete = dp[i - 1][j]       # delete from word1
                    insert = dp[i][j - 1]       # insert into word1
                    dp[i][j] = 1 + min(replace, delete, insert)  # choose min operation
        return dp[m][n]  # minimum operations required
# Time Complexity: O(m * n)
# Space Complexity: O(m * n)
# Algo: DP where each state considers insert, delete, replace operations

# LeetCode 300 – Longest Increasing Subsequence (print sequence)
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        T = [1] * n            # T[i] = length of LIS ending at i
        parent = [-1] * n     # parent[i] = previous index in LIS
        for i in range(n):
            for j in range(i):
                if nums[j] < nums[i] and T[j] + 1 > T[i]:
                    T[i] = T[j] + 1        # update LIS length
                    parent[i] = j          # store predecessor
        lis_len = max(T)                   # length of LIS
        idx = T.index(lis_len)              # index where LIS ends
        seq = []                            # to reconstruct sequence
        while idx != -1:
            seq.append(nums[idx])           # add current element
            idx = parent[idx]               # move to previous element
        seq.reverse()                       # reverse to get correct order
        print(seq)                          # print LIS sequence
        return lis_len                      # return LIS length
# Time Complexity: O(n^2)
# Space Complexity: O(n)
# Algo: DP to compute LIS length, parent array to reconstruct the actual sequence

# LeetCode 152 – Maximum Product Subarray
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        n = len(nums)              # number of elements
        lp = 1                     # prefix product
        rp = 1                     # suffix product
        ans = nums[0]              # maximum product found
        for i in range(n):
            if lp == 0: lp = 1     # reset if zero encountered
            if rp == 0: rp = 1
            lp *= nums[i]          # left to right product
            rp *= nums[n-1-i]      # right to left product
            ans = max(ans, lp, rp) # update answer
        return ans
# Time Complexity: O(n)
# Space Complexity: O(1)

# LeetCode 96 – Unique Binary Search Trees
class Solution:
    def numTrees(self, n: int) -> int:
        # numTree[4] = numTree[0] * numTree[3] +
        #               numTree[1] * numTree[2] +
        #               numTree[2] * numTree[1] +
        #               numTree[3] * numTree[0]
        numTree = [0] * (n + 1)                 # numTree[i] = number of BSTs with i nodes
        numTree[0] = numTree[1] = 1             # base cases: 0 node & 1 node -> 1 BST
        for nodes in range(2, n + 1):            # total nodes in BST
            total = 0                            # count BSTs for current node count
            for root in range(1, nodes + 1):     # choose each node as root
                left = root - 1                  # nodes in left subtree
                right = nodes - root             # nodes in right subtree
                total += numTree[left] * numTree[right]  # combine left & right BSTs
            numTree[nodes] = total               # store result for nodes
        return numTree[n]                        # answer for n nodes
    # # Time Complexity: O(n^2)
    # # Space Complexity: O(n)
    
    # recursion with memo 
    # def numTrees(self, n: int) -> int:
        # memo = {}  # memo[(start, end)] -> number of BSTs in this range
        # def solve(start, end):
        #     if start > end:
        #         return 1                      # empty tree is 1 valid BST
        #     if (start, end) in memo:
        #         return memo[(start, end)]    # reuse computed result
        #     total = 0
        #     for root in range(start, end + 1):      # choose each value as root
        #         left = solve(start, root - 1)       # count left subtrees
        #         right = solve(root + 1, end)        # count right subtrees
        #         total += left * right               # combine left & right
        #     memo[(start, end)] = total               # store result
        #     return total
        # return solve(1, n) 

#   memory limit exceeded
#     def numTrees(self, n: int) -> int:
#         memo = {}  # (start, end) -> list of BST roots
#         def solve(start, end):
#             if start > end:
#                 return [None]                     # empty tree
#             if (start, end) in memo:
#                 return memo[(start, end)]         # reuse result
#             result = []                           # store all BSTs in range
#             for i in range(start, end + 1):       # choose i as root
#                 left_bsts = solve(start, i - 1)   # all left subtrees
#                 right_bsts = solve(i + 1, end)    # all right subtrees
#                 for leftRoot in left_bsts:
#                     for rightRoot in right_bsts:
#                         root = TreeNode(i)        # create root
#                         root.left = leftRoot      # attach left
#                         root.right = rightRoot    # attach right
#                         result.append(root)       # store tree
#             memo[(start, end)] = result            # memoize
#             return result
#         all_trees = solve(1, n)                    # generate all BSTs
#         return len(all_trees)                      # count BSTs
# # Time Complexity: Exponential (Catalan)
# # Space Complexity: Exponential

# LeetCode 516 – Longest Palindromic Subsequence
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        s2 = s[::-1]                         # reverse string
        n = len(s)
        dp = [[0] * (n + 1) for _ in range(n + 1)]  # dp[i][j] = LCS of s[0:i] & s2[0:j]
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if s[i - 1] == s2[j - 1]:    # matching characters
                    dp[i][j] = 1 + dp[i - 1][j - 1]  # extend subsequence
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])  # skip one char
        return dp[n][n]                      # length of LPS
# Time Complexity: O(n^2)
# Space Complexity: O(n^2)
# Logic: Longest Palindromic Subsequence = LCS(s, reverse(s))

# LeetCode 516 – Longest Palindromic Subsequence 
# Time Complexity: O(n^2)
# Space Complexity: O(n^2)
#  without using lcs (Recursion + Memo)
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        n = len(s)
        t = [[-1]*n for _ in range(n)]  # memo table
        def solve(i, j):
            if i > j:
                return 0  # empty range
            if i == j:
                return 1  # single character
            if t[i][j] != -1:
                return t[i][j]  # use memo
            if s[i] == s[j]:
                t[i][j] = 2 + solve(i+1, j-1)  # match ends
            else:
                t[i][j] = max(solve(i+1, j), solve(i, j-1))  # skip one side
            return t[i][j]
        return solve(0, n-1)


# # with LCS
#     def longestPalindromeSubseq(self, s: str) -> int:
#         s2 = s[::-1]                    # reverse string
#         n = len(s)
#         dp = [[0]*(n+1) for _ in range(n+1)]  # dp[i][j] = LCS of s[0:i] and s2[0:j]
#         for i in range(1, n+1):
#             for j in range(1, n+1):
#                 if s[i-1] == s2[j-1]:   # characters match
#                     dp[i][j] = 1 + dp[i-1][j-1]
#                 else:                  # take best by skipping one char
#                     dp[i][j] = max(dp[i-1][j], dp[i][j-1])
#         return dp[n][n]                 # length of LPS
# # Time Complexity: O(n^2)
# # Space Complexity: O(n^2)

# LeetCode 122: Best Time to Buy and Sell Stock II
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n=len(prices)
        if n==0:return 0
        # dp_hold: profit when holding stock
        # dp_free: profit when not holding
        # Example: prices=[7,1,5,3,6,4]
        dp_hold=-prices[0]                  # Day 0: buy at 7 → -7
        dp_free=0                           # Day 0: no stock → 0
        for i in range(1,n):
            # buy or keep holding | Day 1 (1): max(-7,0-1) = -1
            dp_hold=max(dp_hold,dp_free-prices[i])
            # sell or stay free   | Day 2 (5): max(0,-1+5) = 4
            dp_free=max(dp_free,dp_hold+prices[i])
        # Final profit = 7 (1→5, 3→6)
        return dp_free
# Time Complexity: O(n)
# Space Complexity: O(1)

# Minimum sum partition gfg
class Solution:
    def minDifference(self, nums):
        totalSum=sum(nums)                 # total array sum
        n=len(nums)
        k=totalSum
        dp=[[False]*(k+1) for _ in range(n)]  # dp[i][s]: sum s using 0..i
        for i in range(n):
            dp[i][0]=True                 # sum 0 always possible
        if nums[0]<=k:
            dp[0][nums[0]]=True           # first element
        for i in range(1,n):
            for target in range(1,k+1):
                notTake=dp[i-1][target]  # skip element
                take=False
                if nums[i]<=target:
                    take=dp[i-1][target-nums[i]]  # take element
                dp[i][target]=take or notTake
        mini=10**9
        for s1 in range(totalSum//2+1):   # check only half
            if dp[n-1][s1]:
                s2=totalSum-s1
                mini=min(mini,abs(s2-s1))
        return mini
# Time Complexity: O(n * totalSum)
# Space Complexity: O(n * totalSum)

# LeetCode 44: Wildcard Matching
class Solution:
    def isMatch(self, s: str, p: str) -> bool:

        # 1️⃣ Pure Recursion (TLE)
        # def rec(i, j):
        #     if i < 0 and j < 0: return True
        #     if j < 0: return False
        #     if i < 0:
        #         for k in range(j + 1):
        #             if p[k] != '*': return False
        #         return True
        #     if p[j] == s[i] or p[j] == '?':
        #         return rec(i - 1, j - 1)
        #     if p[j] == '*':
        #         return rec(i - 1, j) or rec(i, j - 1)
        #     return False
        # return rec(len(s) - 1, len(p) - 1)
        # Time: Exponential | Space: O(m+n)

        # 2️⃣ Recursion + Memoization (Top-Down DP)
        # m, n = len(s), len(p)
        # dp = [[-1] * n for _ in range(m)]
        # def rec(i, j):
        #     if i < 0 and j < 0: return True
        #     if j < 0: return False
        #     if i < 0:
        #         for k in range(j + 1):
        #             if p[k] != '*': return False
        #         return True
        #     if dp[i][j] != -1: return dp[i][j]
        #     if p[j] == s[i] or p[j] == '?':
        #         dp[i][j] = rec(i - 1, j - 1)
        #     elif p[j] == '*':
        #         dp[i][j] = rec(i - 1, j) or rec(i, j - 1)
        #     else:
        #         dp[i][j] = False
        #     return dp[i][j]
        # return rec(m - 1, n - 1)
        # Time: O(m*n) | Space: O(m*n)

        # 3️⃣ DP Tabulation (Bottom-Up)
        # m, n = len(s), len(p)
        # dp = [[False] * (n + 1) for _ in range(m + 1)]
        # dp[0][0] = True
        # for j in range(1, n + 1):
        #     if p[j - 1] == '*':
        #         dp[0][j] = dp[0][j - 1]
        # for i in range(1, m + 1):
        #     for j in range(1, n + 1):
        #         if p[j - 1] == s[i - 1] or p[j - 1] == '?':
        #             dp[i][j] = dp[i - 1][j - 1]
        #         elif p[j - 1] == '*':
        #             dp[i][j] = dp[i - 1][j] or dp[i][j - 1]
        # return dp[m][n]
        # Time: O(m*n) | Space: O(m*n)

        # 4️⃣ Space Optimized DP (prev / cur)  BEST
        m, n = len(s), len(p)
        prev = [False] * (n + 1)
        cur = [False] * (n + 1)
        prev[0] = True
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                prev[j] = prev[j - 1]
        for i in range(1, m + 1):
            cur[0] = False
            for j in range(1, n + 1):
                if p[j - 1] == s[i - 1] or p[j - 1] == '?':
                    cur[j] = prev[j - 1]
                elif p[j - 1] == '*':
                    cur[j] = prev[j] or cur[j - 1]
                else:
                    cur[j] = False
            prev, cur = cur, [False] * (n + 1)
        return prev[n]
        # Time: O(m*n) | Space: O(n)

# LeetCode 887 – Super Egg Drop
# Idea: DP on number of moves instead of floors

class Solution:
    def superEggDrop(self, k: int, n: int) -> int:
        # dp[m][e] = maximum number of floors that can be tested
        # using m moves and e eggs
        dp = [[0]*(k+1) for _ in range(n+1)]
        moves = 0  # total attempts used
        # keep increasing moves until we can test at least n floors
        while dp[moves][k] < n:
            moves += 1
            for eggs in range(1, k+1):
                # if egg breaks -> dp[m-1][e-1]
                # if egg survives -> dp[m-1][e]
                # +1 for current floor
                dp[moves][eggs] = dp[moves-1][eggs-1] + dp[moves-1][eggs] + 1
        return moves
# Time Complexity: O(k * moves), moves ≈ log(n)
# Space Complexity: O(k * n)
# Algorithm:
# Use DP where each move increases testable floors based on break/survive outcomes.
# Shift focus from minimizing drops per floor to maximizing floors per move.


# TLE
# dp = [[0]*(n+1) for _ in range(k+1)]
# for i in range(1, k+1): # eggs
#     for j in range(1, n+1): # floors
#         if i == 1: 
#             dp[i][j] = j  # max moves = floor 
#         elif j == 1:
#             dp[i][j] = 1 # from floor 1 , moves will be 1 only
#         else:
#             mini = math.inf
#             for x in range(1, j+1):   # drop from floor x
#                 val = max(
#                     dp[i-1][x-1],     # egg breaks
#                     dp[i][j-x]        # egg survives
#                 )
#                 mini = min(mini, val)
#             dp[i][j] = mini + 1
# return dp[k][n]
# Time: O(k · n²) → may TLE for large n
# Space: O(k · n)

# LeetCode 1671 – Minimum Number of Removals to Make Mountain Array
# Approach: Longest Bitonic Subsequence using LIS (left) + LDS (right)
from typing import List
import math
class Solution:
    def minimumMountainRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        LIS = [1]*n  # LIS[i] = longest increasing subsequence ending at i
        LDS = [1]*n  # LDS[i] = longest decreasing subsequence starting at i
        # Compute LIS from left
        for i in range(n):
            for j in range(i):
                if nums[i] > nums[j]:
                    LIS[i] = max(LIS[i], LIS[j] + 1)
        # Compute LDS from right
        for i in range(n-1, -1, -1):
            for j in range(i+1, n):
                if nums[i] > nums[j]:
                    LDS[i] = max(LDS[i], LDS[j] + 1)
        # Find minimum removals by treating each index as peak
        minRemovals = math.inf
        for i in range(n):
            if LIS[i] > 1 and LDS[i] > 1:  # valid mountain peak
                minRemovals = min(minRemovals, n - (LIS[i] + LDS[i] - 1))
        return minRemovals
# Algorithm:
# Build LIS from left and LDS from right.
# For each valid peak, mountain length = LIS[i] + LDS[i] - 1.
# Minimum removals = total elements - maximum mountain length.
# Time Complexity: O(n^2)
# Space Complexity: O(n)

# Matrix Chain Multiplication (MCM)
# Goal: Find minimum scalar multiplications needed to multiply matrices
import math
class Solution:
    def matrixMultiplication(self, arr):
        n = len(arr)
        # ---------------- TABULATION (Bottom-Up DP) ----------------
        # dp[i][j] = minimum cost to multiply matrices from i to j
        dp = [[0]*n for _ in range(n)]
        # fill table by increasing chain length
        for i in range(n-1, 0, -1):
            for j in range(i+1, n):
                mini = math.inf
                for k in range(i, j):
                    # cost = left part + right part + cost of multiplying results
                    steps = arr[i-1]*arr[k]*arr[j] + dp[i][k] + dp[k+1][j]
                    mini = min(mini, steps)
                dp[i][j] = mini
        # final answer: matrices from 1 to n-1
        return dp[1][n-1]

        # ---------------- RECURSION (Top-Down, Exponential) ----------------
        # def f(i, j):
        #     if i == j:  # single matrix, no cost
        #         return 0
        #     mini = math.inf
        #     for k in range(i, j):  # try all partitions
        #         steps = arr[i-1]*arr[k]*arr[j] + f(i, k) + f(k+1, j)
        #         mini = min(mini, steps)
        #     return mini
        # return f(1, n-1)

# Algorithm:
# Try every possible way to parenthesize matrices.
# Choose the partition that minimizes total multiplication cost.

# Time Complexity:
# Recursion: O(2^n)
# Tabulation: O(n^3)
# Space Complexity:
# Recursion: O(n) stack
# Tabulation: O(n^2)

# LeetCode 312 – Burst Balloons
# Approach: Interval DP (Tabulation – Bottom-Up) + Recursion (Commented)

from typing import List
import math

class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        # Add virtual balloons with value 1 at both ends
        nums = [1] + nums + [1]
        n = len(nums) - 2  # original number of balloons
        # dp[i][j] = maximum coins from bursting balloons in range [i, j]
        dp = [[0]*(n+2) for _ in range(n+2)]
        # Bottom-up DP on interval length
        for i in range(n, 0, -1):
            for j in range(i, n+1):
                maxi = 0
                for k in range(i, j+1):  # k is last balloon to burst
                    cost = nums[i-1]*nums[k]*nums[j+1] + dp[i][k-1] + dp[k+1][j]
                    maxi = max(maxi, cost)
                dp[i][j] = maxi
        return dp[1][n]

        # ---------------- RECURSION + MEMO (Top-Down) ----------------
        # nums = [1] + nums + [1]
        # m = len(nums)
        # dp = [[-1]*m for _ in range(m)]
        # def f(i, j):
        #     if i > j: return 0  # no balloons
        #     if dp[i][j] != -1: return dp[i][j]
        #     maxi = 0
        #     for k in range(i, j+1):
        #         cost = nums[i-1]*nums[k]*nums[j+1] + f(i, k-1) + f(k+1, j)
        #         maxi = max(maxi, cost)
        #     dp[i][j] = maxi
        #     return maxi
        # return f(1, m-2)

# Algorithm:
# Treat each balloon as the last to burst in an interval.
# Combine best results from left and right sub-intervals.

# Time Complexity: O(n^3)
# Space Complexity: O(n^2)

