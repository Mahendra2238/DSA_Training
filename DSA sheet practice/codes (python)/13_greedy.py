#1.  455. Assign Cookies
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        i=j=0
        g.sort()
        s.sort()
        while i<len(g) and j<len(s):
            if g[i]<=s[j]:
                i+=1
            j+=1
        return i
g = [1,2,3]
s = [1,1]
sol = Solution()
print(sol.findContentChildren(g,s))

# # 1447. Simplified Fractions
# class Solution:
#     def simplifiedFractions(self, n: int) -> List[str]:
#         res=[]
#         _set=set()
#         for i in range(1,n):
#             for j in range(i+1,n+1):
#                 fraction=i/j
#                 if fraction not in _set:
#                     res.append(str(i)+'/'+str(j))
#                     _set.add(fraction)
#         return res

# 2.fractional knapsack
# **Time Complexity:**
# * Sorting by value/weight ratio: **O(n log n)**
# * Single greedy traversal: **O(n)**
# * **Overall:** **O(n log n)**
# **Space Complexity:**
# * Extra list for `(ratio, value, weight)`: **O(n)**
# * Variables only otherwise: **O(1)**
# * **Overall:** **O(n)**
class Solution:
    def fractionalKnapsack(self, val, wt, capacity):
        n = len(val)
        items = [(val[i] / wt[i], val[i], wt[i]) for i in range(n)]  # (ratio, value, weight)
        items.sort(reverse=True)  # sort by ratio. o(nlogn)
        total_profit = 0.0
        remaining_weight = capacity
        for ratio, value, weight in items:
            if weight <= remaining_weight:  # take full item
                total_profit += value
                remaining_weight -= weight
            else:  # take fraction
                total_profit += ratio * remaining_weight
                break
        return total_profit

# 3. 646. Maximum Length of Pair Chain
# Time Complexity: O(n log n) due to sorting.
# Space Complexity: O(1) as we use only a constant amount of extra space.
class Solution:
    def findLongestChain(self, pairs: List[List[int]]) -> int:
        pairs.sort(key=lambda x: x[1])  # sort by second value
        prev = pairs[0][1]  # first pair
        ans = 1  # chain length
        for i in range(1, len(pairs)):
            if prev < pairs[i][0]:  # valid chain
                prev = pairs[i][1]
                ans += 1
        return ans

# 4. Activity Selection Problem from GFG
# Time Complexity: O(n log n) due to sorting.
# Space Complexity: O(1) as we use only a constant amount of extra space.
class Solution:
    def activitySelection(self, start, finish):
        n = len(start)  # number of activities
        activities = sorted(zip(start, finish), key=lambda x: x[1])  # sort by finish time
        start, finish = zip(*activities)  # unzip start and finish
        i = 1  # next activity index
        j = 0  # last selected activity index
        a = 1  # count first activity
        while i < n:
            if finish[j] < start[i]:  # non-overlapping activity
                a += 1
                j = i  # update last selected
            i += 1
        return a

        # activities = sorted(zip(start, finish), key=lambda x: x[1])  # sort by finish time
        # count = 1  # select first activity
        # last_end = activities[0][1]  # end time of last selected
        # for i in range(1, len(activities)):
        #     if activities[i][0] > last_end:  # check overlap
        #         count += 1  # select activity
        #         last_end = activities[i][1]  # update end time
        # return count

# 5. 1235. Maximum Profit in Job Scheduling
# Time Complexity: O(n log n) due to sorting and binary search.
# Space Complexity: O(n) for the dp array.
class Solution:
    def jobScheduling(self, startTime, endTime, profit):
        n = len(startTime)
        jobs = sorted(zip(startTime, endTime, profit), key=lambda x: x[0])  # sort by start time
        # find first job starting at or after currEnd
        def getNextIndex(l, currEnd):
            r = n - 1
            res = n  # default: no next job
            while l <= r:
                mid = l + (r - l) // 2
                if jobs[mid][0] >= currEnd: # we can make this task
                    res = mid
                    r = mid - 1
                else:
                    l = mid + 1
            return res
        memo = {}
        def solve(i):
            if i >= n:
                return 0
            if i in memo:
                return memo[i]
            next_idx = getNextIndex(i + 1, jobs[i][1])
            take = jobs[i][2] + solve(next_idx)  # take current job
            skip = solve(i + 1)  # skip current job
            memo[i] = max(take, skip)
            return memo[i]
        return solve(0)
# or 
class Solution:
    def jobScheduling(self, startTime, endTime, profit):
        n = len(profit)
        jobs = [[startTime[i], endTime[i], profit[i]] for i in range(n)]
        jobs.sort()  # sort by start time
        start = [jobs[i][0] for i in range(n)]
        dp = [-1] * n  # memo

        def solve(i):
            if i == n:
                return 0
            if dp[i] != -1:
                return dp[i]
            # binary search for next non-overlapping job
            l, r = i + 1, n - 1
            next_idx = n
            while l <= r:
                mid = (l + r) // 2
                if start[mid] >= jobs[i][1]:
                    next_idx = mid
                    r = mid - 1
                else:
                    l = mid + 1
            # take or skip
            take = jobs[i][2] + solve(next_idx)
            skip = solve(i + 1)
            dp[i] = max(take, skip)
            return dp[i]

        return solve(0)
