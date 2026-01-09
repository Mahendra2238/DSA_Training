# class heapp:
#     def __init__(self):
#         self.heap=[]
#     def insert(self,val):
#         self.heap.append(val)
#         self.heapify_up(len(self.heap)-1)
#     def heapify_up(self,i):
#         parent=(i-1)//2
#         while i>0 and self.heap[i]<self.heap[parent]:
#             self.heap[i],self.heap[parent]=self.heap[parent],self.heap[i]        
#             i=parent
#             parent=(i-1)//2
#     def heapify_down(self, i):
#         n = len(self.heap)
#         while True:
#             smallest = i
#             left = 2 * i + 1
#             right = 2 * i + 2

#             if left < n and self.heap[left] < self.heap[smallest]:
#                 smallest = left
#             if right < n and self.heap[right] < self.heap[smallest]:
#                 smallest = right
#             if smallest == i:
#                 break
#             self.heap[i], self.heap[smallest] = self.heap[smallest], self.heap[i]
#             i = smallest
#     def extract_min(self):
#         if len(self.heap) == 0:
#             return None
#         min_val = self.heap[0]
#         self.heap[0] = self.heap.pop()  # Move last to root
#         self.heapify_down(0)
#         return min_val
#     def display(self):
#         print(self.heap)
# h=heapp()
# h.insert(10)
# h.insert(5)
# h.insert(15)
# h.insert(4)
# h.insert(2)
# h.display()
# h.extract_min()
# h.display()
            
# no of unique paths form 0 0 to m-1 n-1
# 1.recursion
# def up(dp,m,n,i,j):
#     if i==m-1 and j==n-1:
#         return 1
#     if i>=m or j>=n:
#         return False
#     return up(dp,m,n,i+1,j) + up(dp,m,n,i,j+1)
# m=7
# n=3
# dp=[[0]*n for _ in range(m)]
# print(up(dp,m,n,0,0))
# Time Complexity: O(2^(m+n))
# Why? At each cell, you branch into two recursive calls.
# Redundant Work: Recomputes the same subproblems many times

# 2.recursive+memorization
# def up(dp,m,n,i,j,memo={}):
#     if i==m-1 and j==n-1:
#         return 1
#     if i>=m or j>=n:
#         return False
#     if (i,j) in memo:
#         return memo[(i,j)]
#     memo[(i,j)]=up(dp,m,n,i+1,j) + up(dp,m,n,i,j+1)
#     return memo[(i,j)]
# m=7
# n=3
# dp=[[0]*n for _ in range(m)]
# print(up(dp,m,n,0,0))
# Time Complexity: O(m × n)
# Why? Each cell (i, j) is computed once and stored in memo.
# Space Complexity: O(m × n) (for the memo dictionary + recursion stack)

# # 3.dp
# def up(m,n):
#     dp=[[1]*n for _ in range(m)]
#     for i in range(1,m):
#         for j in range(1,n):
#             dp[i][j]=dp[i-1][j]+dp[i][j-1]
#     return dp[m-1][n-1]
# m=7
# n=3
# print(up(m,n))
# Time & space Complexity: O(m × n)

# 4.1d
# dp = [1] * n
# for _ in range(1, m):
#     for j in range(1, n):
#         dp[j] += dp[j - 1]
# Time: O(m × n)
# Space: O(n) 

# # 5. Combinatorics Formula
# m=3
# n=2
# import math
# print(math.comb(m + n - 2, m - 1))
# # Time: O(min(m, n)) (for factorial computation)
# # Space: O(1)




