# 39. Combination Sum
# candidates = [2,3,6,7]
# target = 7 
# res=[]
# def backtrack(remaining,comb,ind):
#     if remaining==0:
#         res.append(list(comb))
#         return
#     if remaining<0 or ind==len(candidates):
#         return
#     comb.append(candidates[ind])
#     backtrack(remaining-candidates[ind],comb,ind)
#     comb.pop()
#     backtrack(remaining,comb,ind+1)
# backtrack(target,[],0)
# print(res)

# # 40. Combination Sum II
# candidates = [10,1,2,7,6,1,5]
# target = 8
# candidates.sort()
# res=[]
# def backtrack(target,comb,start):
#      if target==0:
#          res.append(comb[:])
#          return
#      for i in range(start,len(candidates)):
#          if i>start and candidates[i]==candidates[i-1]: 
#              continue
#          if candidates[i]>target:
#              continue
#          comb.append(candidates[i])
#          backtrack(target-candidates[i],comb,i+1)
#          comb.pop()
# backtrack(target,[],0)
# print(res)

# 131. Palindrome Partitioning
# s="aab"
# ans=[]
# def getAllParts(s,partitions):
#     if not s: #len(s)==0:
#         ans.append(list(partitions))
#         return
#     for i in range(0,len(s)):
#         part=s[0:i+1]
#         if part==part[::-1]: # if isPalin(part):
#             partitions.append(part)
#             getAllParts(s[i+1:],partitions)
#             partitions.pop()
# getAllParts(s,[])
# print(ans)

# 2596. Check Knight Tour Configuration
# grid = [[0,11,16,5,20],[17,4,19,10,15],[12,1,8,21,6],[3,18,23,14,9],[24,13,2,7,22]]
# def isValid(grid,r,c,n,expVal): # t.c: 8^n^2    s.c: n^2
#     if r<0 or c<0 or r>=n or c>=n or grid[r][c]!=expVal:
#         return False 
#     if expVal==n*n-1: #last val (0 to n*n-1 in grid) 
#         return True
#     # possible moves (8)
#     ans1 = isValid(grid,r-2,c+1,n,expVal+1)
#     ans2 = isValid(grid,r-1,c+2,n,expVal+1)
#     ans3 = isValid(grid,r+1,c+2,n,expVal+1)
#     ans4 = isValid(grid,r+2,c+1,n,expVal+1)
#     ans5 = isValid(grid,r+2,c-1,n,expVal+1)
#     ans6 = isValid(grid,r+1,c-2,n,expVal+1)
#     ans7 = isValid(grid,r-1,c-2,n,expVal+1)
#     ans8 = isValid(grid,r-2,c-1,n,expVal+1)
#     return ans1 or ans2 or ans3 or ans4 or ans5 or ans6 or ans7 or ans8
# print(isValid(grid,0,0,len(grid),0))

# M-Coloring Problem (gfg)
# def graphColoring(v, edges, m):
#     # code here
#     adj=[[] for _ in range(v)]
#     for x,y in edges:
#         adj[x].append(y)
#         adj[y].append(x)
#     colors=[0]*v
#     def isSafe(node,c):
#         for nei in adj[node]:
#             if colors[nei]==c:
#                 return False
#         return True
#     def solve(node):
#         if node==v:
#             return True
#         for c in range(1,m+1):
#             if isSafe(node,c):
#                 colors[node]=c
#                 if solve(node+1): return True
#                 colors[node]=0
#         return False
#     return solve(0)
# v = 4
# edges = [[0, 1], [1, 3], [2, 3], [3, 0], [0, 2]]
# m = 3
# print(graphColoring(v, edges, m))

# Rat in Maze 
# Time	O(m^V · V)
# Space	O(V + E) (or O(V) extra)
# # Function to find all possible paths
# def ratInMaze(maze):
#     def paths(i,j,p,n):
#         if i<0 or j<0 or i>n-1 or j>n-1 or maze[i][j]!=1:
#             return
#         if i==n-1 and j==n-1:
#             res.append(p)
#             return 
#         maze[i][j]=-1
#         paths(i+1,j,p+'D',n)
#         paths(i,j-1,p+'L',n)
#         paths(i,j+1,p+'R',n)
#         paths(i-1,j,p+'U',n)
#         maze[i][j]=1
#     # code here
#     n=len(maze)
#     res=[]
#     if maze[0][0]==1:
#         paths(0,0,"",n)
#     return res
# maze = [[1, 0, 0, 0], [1, 1, 0, 1], [1, 1, 0, 0], [0, 1, 1, 1]]
# print(ratInMaze(maze))

# # 90. Subsets II
# nums = [1,2,2]
# nums.sort()
# allsubsets=[]
# def getAllSubsets(idx,ss):
#     if idx==len(nums):
#         allsubsets.append(ss[:])
#         return
#     # include
#     ss.append(nums[idx])
#     getAllSubsets(idx+1,ss)
#     ss.pop()
#     i=idx+1
#     while i<len(nums) and nums[i]==nums[i-1]:
#         i+=1
#     # exclude
#     getAllSubsets(i,ss)
# getAllSubsets(0,[])
# print(allsubsets)

# 912. Sort an Array
# tc=o(nlogn)
# sc=o(n)+o(n) (arr space + recursion stack)
# nums = [5,2,3,1]
# def merge(nums,s,m,e):
#     n1=m-s+1
#     n2=e-m  # e-(m+1)+1 = e-m-1+1 = e-m
#     arr1=nums[s:m+1]
#     arr2=nums[m+1:e+1]
#     i=0
#     j=0
#     k=s # starting at s
#     while i<n1 and j<n2:
#         if arr1[i]<=arr2[j]:
#             nums[k]=arr1[i]
#             i+=1
#         else:
#             nums[k]=arr2[j]
#             j+=1
#         k+=1
#     while i<n1:
#         nums[k]=arr1[i]
#         i+=1
#         k+=1
#     while j<n2:
#         nums[k]=arr2[j]
#         j+=1
#         k+=1
# def mergeSort(nums,s,e):
#     if s<e:
#         m=s+(e-s)//2
#         mergeSort(nums,s,m)
#         mergeSort(nums,m+1,e)
#         merge(nums,s,m,e)
# mergeSort(nums,0,len(nums)-1)
# print(nums)

# 51. N-Queens
# n=4
# def isSafe(board,row,col,n):
#     # Horizontal
#     for j in range(n):
#         if board[row][j]=='Q':
#             return False
#     # Vertical
#     for i in range(n):
#         if board[i][col]=='Q':
#             return False
#     # Left diagonal (up) \
#     i,j=row,col
#     while i>=0 and j>=0:
#         if board[i][j]=='Q':
#             return False
#         i-=1
#         j-=1
#     # Right diagonal /
#     i,j=row,col
#     while i>=0 and j<n:
#         if board[i][j]=='Q':
#             return False
#         i-=1
#         j+=1
#     return True
# def nQueens(board,row,n):
#     if row==n:
#         ans.append(["".join(r) for r in board])
#         return    
#     for col in range(n):
#         if isSafe(board,row,col,n):
#             board[row][col]='Q'
#             nQueens(board,row+1,n)
#             board[row][col]='.'
# ans=[]
# board=[['.']*n for _ in range(n)]
# nQueens(board,0,n)
# print(ans)

# 37. Sudoku Solver
# class Solution:
#     def solveSudoku(self, board):
#         """
#         Do not return anything, modify board in-place instead.
#         """
#         rows=[set() for _ in range(9)]
#         cols=[set() for _ in range(9)]
#         boxes=[set() for _ in range(9)]
#         for i in range(9):
#             for j in range(9):
#                 if board[i][j]!='.':
#                     num=board[i][j]
#                     rows[i].add(num)
#                     cols[j].add(num)
#                     boxes[(i//3)*3+(j//3)].add(num)
#         def isSafe(row,col,dig):
#             if dig in rows[row]:return False
#             if dig in cols[col]:return False
#             if dig in boxes[(row//3)*3+(col//3)]:return False
#             return True
#         def helper(row,col):
#             if row==9:return True
#             if col==8:
#                 nextRow,nextCol=row+1,0
#             else:
#                 nextRow,nextCol=row,col+1
#             if board[row][col]!='.':
#                 return helper(nextRow,nextCol)
#             for dig in "123456789":
#                 if isSafe(row,col,dig):
#                     board[row][col]=dig
#                     rows[row].add(dig)
#                     cols[col].add(dig)
#                     boxes[(row//3)*3+(col//3)].add(dig)
#                     if helper(nextRow,nextCol):
#                         return True
#                     board[row][col]='.'
#                     rows[row].remove(dig)
#                     cols[col].remove(dig)
#                     boxes[(row//3)*3+(col//3)].remove(dig)
#             return False
#         helper(0,0)
# board = [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]
# Solution().solveSudoku(board)
# print(board)
#         # # def isSafe(board,row,col,dig):
#         # #     # check row - only need to check current row
#         # #     for j in range(9):
#         # #         if board[row][j]==dig:
#         # #             return False
#         # #     # check col - only need to check current column
#         # #     for i in range(9):
#         # #         if board[i][col]==dig:
#         # #             return False
#         # #     # check grid
#         # #     sr=(row//3)*3  
#         # #     sc=(col//3)*3
#         # #     for i in range(sr,sr+3):
#         # #         for j in range(sc,sc+3):
#         # #             if board[i][j]==dig:
#         # #                 return False
#         # #     return True
        
#         # def isSafe(board,row,col,dig):
#         #     for k in range(len(board)):
#         #         if board[row][k]==dig:
#         #             return False
#         #         if board[k][col]==dig:
#         #             return False
#         #         if board[3*(row//3)+k//3][3*(col//3)+k%3]==dig:
#         #             return False
#         #     return True
#         # def helper(board,row,col):
#         #     # Base case: if we've processed all rows
#         #     if row == 9:
#         #         return True
#         #     # Calculate next position
#         #     if col == 8:
#         #         nextRow = row + 1
#         #         nextCol = 0
#         #     else:
#         #         nextRow = row
#         #         nextCol = col + 1
#         #     # If current cell is already filled
#         #     if board[row][col] != '.':
#         #         return helper(board,nextRow,nextCol)
#         #     # Try placing digits 1-9
#         #     for dig in "123456789":
#         #         if isSafe(board,row,col,dig):
#         #             board[row][col]=dig
#         #             if helper(board,nextRow,nextCol):
#         #                 return True
#         #             # Backtrack
#         #             board[row][col]='.'
#         #     return False
#         # helper(board,0,0)