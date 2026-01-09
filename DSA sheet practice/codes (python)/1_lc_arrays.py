# ARRAYS 

# 1. pair sum  (arr is sorted)

# Brute force method o(n^2)
# take all pairs and check sum is equal to target
# arr=[2,7,11,15]
# target=9
# for i in range(len(arr)):
#     for j in range(i+1,len(arr)):
#         if arr[i]+arr[j]==target:
#             print(i,j)
#             break

# Two pointer o(n)
# arr=[2,7,11,15]
# target=9
# l=0
# r=len(arr)-1
# while l<r:
#     ps=arr[l]+arr[r]
#     if ps<target:
#         l+=1
#     elif ps>target:
#         r-=1
#     else:
#         print(l,r)
#         break

# 2. Majority element (if freq is > n/2) floor-> 3.4 =3

# Bruteforce o(n2)
# a=[1,2,2,1,1]
# n=len(a)
# for i in range(n):
#     freq=0
#     for j in range(n):
#         if a[i]==a[j]:
#             freq+=1
#     if freq>n/2:
#         print(a[i])
#         break

# optimal o(nlogn)   // o(nlogn+n) nlogn for sorting n for loop
# a=[1,2,2,1,1]
# n=len(a)
# a.sort()
# freq=1
# ans=a[0]
# for i in range(1,n):
#     if a[i]==a[i-1]:
#         freq+=1
#     else:
#         freq=1
#         ans=a[i]
#     if freq>n/2:
#         print(ans)
#         break

# Moores voting algorithm o(n)
# same ele -> freq++ , diff ele -> freq-- 
# a=[1,2,2,1,1]
# n=len(a)
# freq,ans=0,0
# for i in range(n):
#     if freq==0:
#         ans=a[i]
#     if ans==a[i]:
#         freq+=1
#     else:
#         freq-=1
# c=0 # check if ans is majority eg: [1,2,3,4] 4 is not majority
# for k in a:
#     if k==ans:
#         c+=1
# if c>n/2:
#     print(ans)
# else:
#     print(-1)

# o(n) o(1)
# from collections import Counter
# def majorityElement(nums):
#     n=len(nums)
#     # c=Counter(nums)
#     # for k,v in c.items():
#     #     if v>n/2:
#     #         return k
#     nums.sort()
#     return nums[n//2]
# a=[1,2,2,1,1]
# print(majorityElement(a))

# # 3. 2965. Find Missing and Repeated Values
# from collections import Counter
# def findMissingAndRepeatedValues(grid):
#     n=len(grid)
#     size=n*n
#     count=[0]*(size+1)
#     for i in range(n):
#         for j in range(n):
#             count[grid[i][j]]+=1
#     a,b=-1,-1
#     for k in range(1,size+1):
#         if count[k]==2:
#             a=k
#         elif count[k]==0:
#             b=k
#     return [a,b]
#     # vi=set()
#     # for i in range(len(grid)):
#     #     for j in range(len(grid[0])):
#     #         if grid[i][j] not in vi:
#     #             vi.add(grid[i][j])
#     #         else:
#     #             rn=grid[i][j]
#     # t=sum(vi)
#     # n=len(grid)*len(grid[0])
#     # ts=n*(n+1)//2
#     # mn=ts-t
#     # return ([rn,mn])
# grid = [[1,3],[2,2]]
# print(findMissingAndRepeatedValues(grid))

# o(n2)
# grid = [[9,1,7],[8,9,2],[3,4,6]]
# n=len(grid)
# vi=set()
# act_sum=0
# for i in range(n):
#     for j in range(n):
#         act_sum+=grid[i][j]
#         if grid[i][j] in vi:
#             rn=grid[i][j]
#         vi.add(grid[i][j])
# exp_sum=(n*n)*(n*n+1)//2
# mn=exp_sum+rn-act_sum
# print(rn,mn)

# # 4. 2 sum
# # brute force - o(n2)
# # better- twopointer - o(nlogn)
# # optimal - hashing - o(n)
# from collections import defaultdict
# a=[5,2,11,7,15]
# tar=9
# n=len(a)
# d=defaultdict(int)
# for i in range(n):
#     f=a[i]
#     s=tar-f
#     if s in d:
#         print([i,d[s]])
#         break
#     d[f]=i
    
# 5. Find duplicates
# tc - o(n)
# sc - o(n) due to set
# a=[1,3,4,2,2]
# vi=set()
# for i in a:
#     if i in vi:
#         print(i)
#     vi.add(i)
# print(vi)

# tc - o(n)
# sc - o(1) - using fast and slow pointer
# a=[3,1,3,4,2]
# s=f=a[0]
# while True:     # 0 3 4 2
#     s=a[s] #+1      
#     f=a[a[f]] #+2
#     if s==f:
#         break
# s=a[0]
# while s!=f: 
#     s=a[s] #+1
#     f=a[f] #+1
# print(s) # s or f

# 6. 88. Merge Sorted Array o(n+m)
# You are given two sorted integer arrays. Merge nums1 and nums2 into a single array sorted in non-decreasing order.
# The final sorted array should be stored inside the array nums1. 
# nums1=[1,2,3,0,0,0]
# nums2=[2,5,6]
# m,n=3,3
# idx=m+n-1
# i=m-1
# j=n-1
# # while i>=0 and j>=0:
# while j>=0:
#     if nums1[i]>nums2[j] and i>=0:
#         nums1[idx]=nums1[i]
#         i-=1
#     else:
#         nums1[idx]=nums2[j]
#         j-=1
#     idx-=1
# # while j>=0:
# #     nums1[idx]=nums2[j]
# #     j-=1
# #     idx-=1
# print(nums1)

# 7. 31. Next Permutation (The replacement must be in place and use only constant extra memory.)
# For example, for arr = [1,2,3], the following are all the permutations of arr: [1,2,3], [1,3,2], [2, 1, 3], [2, 3, 1], [3,1,2], [3,2,1].
# the next permutation of arr = [2,3,1] is [3,1,2]
# nums = [1,2,3]
# # nums=[3,2,1]
# n=len(nums)
# p=-1
# i=n-1
# # s1: find the pivot
# while i>0 and nums[i-1]>=nums[i]: # 12(3)54
#     i-=1
# p=i-1
# if p==-1:  # if all are in decreasing just reverse eg: 54321 -> 12345
#     nums.reverse()
#     print(nums)
#     exit()
#     # return
# # s2: swap pivot and next larger element  12(4)53
# for i in range(n-1,-1,-1):
#     if nums[i]>nums[p]:
#         nums[i],nums[p]=nums[p],nums[i]
#         break
# # s3: reverse (p+1 to n-1)  12435   so we got next permutation of 12354 is 12435
# nums[p+1:]=reversed(nums[p+1:n])
# # i,j=p+1,n-1
# # while i<j:
# #     nums[i],nums[j]=nums[j],nums[i]
# #     i+=1
# #     j-=1
# print(nums)

# 8. 136. Single Number 
# Find that single one. You must implement a solution with a linear runtime complexity and use only constant extra space
# Bitmanipulation
# tc=o(n) sc=o(1) 
# nums = [4,1,2,1,2]  # (n^n=0, n^0=n)   (4 +1 +2 -1 -2 = 4)
# ans=0
# for i in range(len(nums)):
#     ans^=nums[i]
# print(ans)

# tc=o(n) sc=o(1)
# from collections import Counter
# c = Counter(nums)
# for i in nums:
#     if c[i] == 1:
#         print(i)
#         break

# # 9. 50. Pow(x, n)
# # Constraints:
# # -100.0 < x < 100.0
# # -2^31 <= n <= 2^31-1
# # 10^8 ops < n=2^31  so we compute in binary form
# # 8->1000  -> logn+1 = 3+1=4 digits (4 operations/times) , 105->3  
# def powr(x,n):
#     if n==0: return 1.0
#     if x==0: return 0.0
#     if x==1: return 1.0
#     if x==-1 and n%2==0: return 1.0   # power even
#     if x==-1 and n%2!=0: return -1.0  # pow odd
    
#     bnf=n   # if pow is negative eg: 2^-2 = 1/2^2 = 1/4 = 0.25
#     if n<0:
#         x=1/x
#         bnf=-bnf # -(-bnf)=bnf
#     #  (3^5=3*3*3*3*3 o(n) linear)    101=> 1--3^4. 0--3^2. 1--3^1 --> 3^1.3^4=3^5  (3,9,81,243)  x16<-x8<-x4<-x2<-x1   (x1.x1=x2  x2.x2=x4 x4.x4=x8...)  
#     ans=1
#     while bnf>0:
#         if bnf%2==1: # when last digit is 1 
#             ans*=x  # 1*3^1*3^4 = 3^5= 243
#         x=x*x # 9 81 243
#         bnf//=2 # 5(101) 2(10) 1(1) 0 
#     return ans
# x = 2.00000
# n = -2
# print(powr(x,n))

# 10. 121. Best Time to Buy and Sell Stock
# prices = [7,1,5,3,6,4]
# maxprof=0
# minbuy=prices[0]
# for i in range(1,len(prices)):
#     minbuy=min(minbuy,prices[i])
#     maxprof=max(maxprof,prices[i]-minbuy)
# print(maxprof)

# 11. Max Subarray Sum
# o(n2)
# arr=[-2,1,-3,4,-1,2,1,-5,4]
# n=len(arr)
# ms=-9999
# for i in range(n):
#     cs=0
#     for j in range(i,n):
#         cs+=arr[j]
#         ms=max(ms,cs)
# print(ms)

# # Kadanes algorithm  o(n)    when cs is negative make cs=0 as ms can't be less than zero  eg. 3,-4,5,4..   when 3+(-4)=-1 make cs=0 as eliminate 3,-4 as it cant give ms
# arr=[5,4,-1,7,8]
# n=len(arr)
# cs=0
# ms=-9999
# for i in range(n):
#     cs+=arr[i]
#     ms=max(ms,cs)
#     if cs<0:  # down, because if all negative [-1,-2,-3] to handle this edge case
#         cs=0
# print(ms)

# # 12. 11. Container With Most Water
# # bruteforce o(n2)
# height = [1,8,6,2,5,4,8,3,7]
# cw,mw=0,0
# for i in range(len(height)):
#     for j in range(i+1,len(height)):
#         w=j-i
#         h=min(height[i],height[j])
#         cw=w*h
#         mw=max(mw,cw)
# print(mw)
# # two pointer approach o(n)
# height = [1,8,6,2,5,4,8,3,7]
# l,r=0,len(height)-1
# cw,mw=0,0
# while l<r:
#     w=r-l
#     h=min(height[l],height[r])
#     cw=w*h
#     mw=max(mw,cw)
#     if height[l]<height[r]:
#         l+=1   
#     else:
#          r-=1
# print(mw)

# 13. Trapping rain water
# prefix array  t.c=o(n) s.c=o(n) (auxillary space)
# h=[4,2,0,3,2,5]
# n=len(h)
# lh=[0]*n
# rh=[0]*n
# lmax,rmax=h[0],h[n-1]
# for i in range(n):
#     if lmax<h[i]:
#         lmax=h[i]
#     lh[i]=lmax
# for j in range(n-1,-1,-1):
#     if rmax<h[j]:
#         rmax=h[j]
#     rh[j]=rmax
# # print(lh,rh)
# w=0
# for k in range(n):
#     w+=min(lh[k],rh[k])-h[k]
# print(w)

# Two pointer approach tc=o(n) sc=o(1)
# h=[4,2,0,3,2,5]
# n=len(h)
# l,r=0,n-1
# lmax=rmax=w=0
# while l<r:
#     lmax=max(lmax,h[l])
#     rmax=max(rmax,h[r])
#     if lmax<rmax:
#         w+=lmax-h[l]
#         l+=1
#     else:
#         w+=rmax-h[r]
#         r-=1
# print(w)

# 14. 75. Sort Colors (sort them in-place so that objects of the same color are adjacent. We will use the integers 0, 1, and 2 to represent the color red, white, and blue, respectively.)

# 1.bruteforce o(n2) -> sort
# 2.optimized o(n) 
# from collections import defaultdict
# nums = [2,0,2,1,1,0]
# d=defaultdict(int)
# for i in nums:
#     d[i]+=1
# idx=0
# for i in range(3):
#     f=d[i]
#     nums[idx:idx+f]=[i]*f
#     idx=idx+f
# print(nums)

# 3. optimal o(n) single pass   
# Dutch National Flag (DNf) algorithm (3 pointers-> l,m,h)
# 0 to l      -> 0
# l+1 to m-1  -> 1
# m to h      -> unsorted ele
# h+1 to n-1  -> 2
# nums = [2,0,2,1,1,0]
# l=m=0
# h=len(nums)-1
# while m<=h:
#     if nums[m]==0:
#         nums[l],nums[m]=nums[m],nums[l]
#         l+=1
#         m+=1
#     elif nums[m]==1:
#         m+=1
#     else:
#         nums[m],nums[h]=nums[h],nums[m]
#         h-=1
# print(nums)

# 15. 3sum

# bruteforce
# Time: O(n³) 
# Space: O(k) — number of unique triplets
# nums = [-1,0,1,2,-1,-4]
# l=set()
# n=len(nums)
# for i in range(n):
#     for j in range(i+1,n):
#         for k in range(j+1,n):
#             s=nums[i]+nums[j]+nums[k]
#             if s==0:
#                 l.add(tuple(sorted([nums[i],nums[j],nums[k]])))
# print([list(t) for t in l])

# # better+hashing
# Time: O(n²)
# Space: O(n) extra space for the hash set per outer loop
# # from collections import defaultdict
# nums = [-1,0,1,2,-1,-4]
# n=len(nums)
# res=set()
# for i in range(n):
#     a=nums[i]
#     seen=set()
#     for j in range(i+1,n):
#         b=nums[j]
#         c=-a-b
#         if c in seen:
#             res.add(tuple(sorted((a,b,c))))
#         seen.add(b)
# print([list(t) for t in res])

# optimal (two pointer)
# tc=o(n2) sc=o(1)
# a = [-1,0,1,2,-1,-4]
# a.sort()
# n=len(a)
# res = []
# for i in range(n):
#     if i>0 and a[i]==a[i-1]:
#         continue
#     j=i+1
#     k=n-1
#     while j<k:
#         s=a[i]+a[j]+a[k]
#         if s<0:
#             j+=1
#         elif s>0:
#             k-=1
#         else:
#             res.append([a[i],a[j],a[k]])
#             j+=1
#             k-=1
#             while j<k and a[j]==a[j-1]:
#                 j+=1
#             while j<k and a[k]==a[k+1]:
#                 k-=1
# print(res)
    
# # 16. 4sum
# # Time	O(n³)
# # Space	O(1) (excluding output)
# # Output Space	O(k) where k is number of unique quadruplets
# arr = [1,0,-1,0,-2,2]
# target = 0
# n = len(arr)
# arr.sort() 
# l=[]
# for i in range(n - 3):
#     if i > 0 and arr[i] == arr[i - 1]:  
#         continue
#     for j in range(i + 1, n - 2):
#         if j > i + 1 and arr[j] == arr[j - 1]: 
#             continue
#         left = j + 1
#         right = n - 1
#         while left < right:
#             current_sum = arr[i] + arr[j] + arr[left] + arr[right]
#             if current_sum == target:
#                 l.append([arr[i],arr[j],arr[left],arr[right]])
#                 while left < right and arr[left] == arr[left + 1]:
#                     left += 1
#                 while left < right and arr[right] == arr[right - 1]:
#                     right -= 1
#                 left += 1
#                 right -= 1
#             elif current_sum < target:
#                 left += 1
#             else:
#                 right-=1
# print(l)

# 17. 74. Search a 2D Matrix (O(log(m * n)))
# Each row is sorted in non-decreasing order.
# The first integer of each row is greater than the last integer of the previous row.
# mat = [[1,3,5,7],[10,11,16,20],[23,30,34,60]]
# target = 3
# r,c=len(mat),len(mat[0])  
# l=0
# h=(r*c)-1 #3*4=12 -1=11
# f=False
# while l<=h:
#     m=(l+h)//2
#     if mat[m//c][m%c]==target:
#         f=True
#         break
#     if mat[m//c][m%c]<target:
#         l=m+1
#     else:
#         h=m-1
# print(f)
#  (or)
# o(logm +logn)=o(logmn)
# def bs(mat,t,r,c):
#     l=0
#     h=c-1
#     while l<=h:
#         m=l+h//2
#         if mat[r][m]==t:
#             return True
#         if mat[r][m]>t:
#             l=m+1
#         else:
#             h=m-1
#     return False        
# mat = [[1,3,5,7],[10,11,16,20],[23,30,34,60]]
# t = 3
# r,c=len(mat),len(mat[0])
# srow=0
# erow=r-1
# f=False
# while srow<=erow:
#     mrow=srow+(erow-srow)//2
#     if t>=mat[mrow][0] and t<=mat[mrow][c-1]:
#         f=bs(mat,t,mrow,c)
#         break
#     if t>mat[mrow][c-1]:
#         srow=mrow+1
#     else:
#         erow=mrow-1
# print(f)
    
# 18.240. Search a 2D Matrix II  o(m+n)
# Integers in each row are sorted in ascending from left to right.
# Integers in each column are sorted in ascending from top to bottom.
# mat = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]]
# k=5
# r=0
# c=len(mat[0])-1
# f=False
# while r<len(mat) and c>=0:
#     if k==mat[r][c]:
#         f=True
#         break
#     if k<mat[r][c]:
#         c-=1
#     else:
#         r+=1
# print(f)

# # 19, 56. Merge Intervals
# # merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input
# intervals = [[1,3],[2,6],[8,10],[15,18]]
# intervals.sort()
# mi=[]
# cur=intervals[0]
# for ivl in intervals[1:]:
#     if cur[1]>=ivl[0]:
#         cur[1]=max(ivl[1],cur[1])
#     else:
#         mi.append(cur)
#         cur=ivl
# mi.append(cur)    
# print(mi)

# 20. 3. Longest Substring Without Repeating Characters o(n) sliding window
# s = "pwwkew"
# l=0
# maxlen=0
# cs=set()
# for r in range(len(s)):
#     while s[r] in cs:        
#         cs.remove(s[l])
#         l+=1
#     cs.add(s[r])
#     maxlen=max(maxlen,(r-l)+1)
# print(maxlen)

# 21. 73. Set Matrix Zeroes (inplace)
# Bruteforce o (nxm)x(n+m)+(nxm)  (somewhere like o(n3))  sc=o(1)
# def makerow(r):
#     for c in range(n):
#         if mat[r][c]!=0:
#             mat[r][c]=-1
# def makecol(c):
#     for r in range(m):
#         if mat[r][c]!=0:
#             mat[r][c]=-1
# mat = [[1,1,1],[1,0,1],[1,1,1]]
# m,n=len(mat),len(mat[0])
# for i in range(m):
#     for j in range(n):
#         if mat[i][j]==0:
#             makerow(i)
#             makecol(j)
# for i in range(m):
#     for j in range(n):
#         if mat[i][j]==-1:
#             mat[i][j]=0
# print(mat)

# Better
# t.c=o(2xnxm) sc=o(n+m)
# matrix = [[1,1,1],[1,0,1],[1,1,1]]
# m,n=len(matrix),len(matrix[0])
# row=[0]*m
# col=[0]*n
# for i in range(m):
#     for j in range(n):
#         if matrix[i][j]==0:
#             row[i]=1
#             col[j]=1
# for i in range(m):
#     for j in range(n):
#         if row[i]==1 or col[j]==1:
#             matrix[i][j]=0
# print(matrix)

# tc=o(m*n) s=o(1) [optimal] 
# matrix = [[1,1,1],[1,0,1],[1,1,1]]
# r,c=len(matrix),len(matrix[0])
# col0=1
# for i in range(r):
#     for j in range(c):
#         if matrix[i][j]==0:
#             matrix[i][0]=0
#             if j!=0:
#                 matrix[0][j]=0
#             else:
#                 col0=0
# for i in range(1,r):
#     for j in range(1,c):
#         if matrix[i][j]!=0:
#             if matrix[i][0]==0 or matrix[0][j]==0:
#                 matrix[i][j]=0
# if matrix[0][0]==0:
#     for j in range(c):
#         matrix[0][j]=0
# if col0==0:
#     for i in range(r):
#         matrix[i][0]=0
# print(matrix)

# 22. 79. Word Search (backtracking ,dfs, array)
# Return true if word exists in the grid. The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once.
# tc= O(m × n × 4^L), where:
# m × n = board size
# L = length of word
# sc= O(L) recursion depth for DFS
# def find(i,j,idx):
#     if idx==len(word):
#         return True
#     if i<0 or j<0 or i>=m or j>=n or board[i][j]!=word[idx] or board[i][j]=='$': # $=visited
#         return False
#     tmp=board[i][j]
#     board[i][j]='$'
#     if find(i+1,j,idx+1) or find(i-1,j,idx+1) or find(i,j+1,idx+1) or find(i,j-1,idx+1):
#         return True
#     board[i][j]=tmp
#     return False
# board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
# word = "ABCCED"
# m,n=len(board),len(board[0])
# f=0
# for i in range(m):
#     for j in range(n):
#         if board[i][j]==word[0] and find(i,j,0):
#             f=1
# print(False if f==0 else True)

        # answer=[1]*len(nums)
        # for i in range(len(nums)):
        #     for j in range(len(nums)):
        #         if i!=j:
        #           answer[i]*=nums[j]
        # return answer

# 23. 238. Product of Array Except Self
# bruteforce o(n2)
# nums = [1,2,3,4]
# answer=[1]*len(nums)
# for i in range(len(nums)):
#     for j in range(len(nums)):
#         if i!=j:
#           answer[i]*=nums[j]
# print(answer)

# better o(3n)-> o(n)  sc=o(n)
# nums = [1,2,3,4]
# ans=[0]*len(nums)  
# prefix=[1]*len(nums)
# suffix=[1]*len(nums)
# for i in range(1,len(nums)):
#     prefix[i]=prefix[i-1]*nums[i-1]
# for i in range(len(nums)-2,-1,-1):
#     suffix[i]=suffix[i+1]*nums[i+1]
# for i in range(len(nums)):
#     ans[i]=prefix[i]*suffix[i]
# print(ans)

# optimal o(n2) o(1)
# nums = [1,2,3,4] # eg: 1 2 (3) 4 5 for index at 3 mul prefixprod(left) and suffixprod(right) of 3
# ans=[1]*len(nums)  #[1,1,1,1]
# #prefix prod
# for i in range(1,len(nums)):
#     ans[i]=ans[i-1]*nums[i-1] # [1,1,2,6]
# #suffix prod
# suffix=1
# for i in range(len(nums)-2,-1,-1):
#     suffix*=nums[i+1]  #suf=[24,12,4,1]
#     ans[i]*=suffix  # [24,12,8,6]    
# print(ans)

# 24.subarraysum equals K
# bruteforce o(n2)
# nums =[1,2,3]
# k = 3
# c=0
# for i in range(len(nums)):
#     s=0
#     for j in range(i,len(nums)):
#         s+=nums[j]
#         if s==k:
#             c+=1
# print(c)

# optimal tc=o(n) sc=o(n) for hashmap
# from collections import defaultdict
# nums =[1,2,3]
# k = 3
# count = 0
# s = 0
# hm = defaultdict(int)
# hm[0] = 1  # Crucial for handling subarrays that start at index 0
# print(hm)
# for num in nums:
#     s += num
#     count += hm[s - k] #ps[j]-ps[i-1]=k we use hash map bcoz o(1) insted of o(n) list and if ps=[1,13,13,..] if ps is same count freq
#     hm[s] += 1
# print(count)

# 25. Counting inversions (Given an array , return the number of inversions to sort the array.) hackerrank
# [2,4,1]->The sort has two inversions: (4,1)  and (2,1).
# bruteforce o(n2)
# arr=[6,3,5,2,7]
# # arr=[2,4,1]
# n=len(arr)
# ic=0
# for i in range(n):
#     for j in range(i+1,n):
#         if arr[i]>arr[j]:
#             ic+=1
# print(ic)

# # optimal (mergesort) tc=o(nlogn) sc=o(n)
# def merge(arr,l,m,h):
#     i=l
#     j=m+1
#     k=0
#     ic=0
#     ctmp=[0]*(h-l+1) #len(arr)
#     while i<=m and j<=h:
#         if arr[i]<arr[j]:
#             ctmp[k]=arr[i]
#             i+=1
#         else:
#             ctmp[k]=arr[j]
#             j+=1
#             ic+=(m-i+1)
#         k+=1
#     while i<=m:
#         ctmp[k]=arr[i]
#         i+=1
#         k+=1
#     while j<=h:
#         ctmp[k]=arr[j]
#         j+=1
#         k+=1
#     for idx in range(len(ctmp)):
#         arr[l+idx]=ctmp[idx]
#     return ic
# def mergesort(arr,l,h):
#     if l<h:
#         m=(l+h)//2
#         lc=mergesort(arr,l,m)
#         rc=mergesort(arr,m+1,h)
#         invc=merge(arr,l,m,h)
#         return lc+rc+invc
#     return 0
# arr=[6,3,5,2,7]
# n=len(arr)
# ict=mergesort(arr,0,n-1)
# print(ict)

# 26. 239. Sliding Window Maximum
# Given an array `nums` and window size `k`, slide a window from left to right, one step at a time. At each step, return the **maximum value** within the current window.
# bruteforce o(n*k)
# nums = [1,3,-1,-3,5,3,6,7]
# k = 3
# n=len(nums)
# res=[]
# for i in range(n-k+1):
#     mx=0
#     for j in range(i,i+k):
#         mx=max(mx,nums[j])
#     res.append(mx)
# print(res)

# optimal tc=o(n) sc=o(n)
# from collections import deque
# nums = [1,3,-1,-3,5,3,6,7]
# n=len(nums)
# k = 3
# dq=deque()
# res=[]
# for i in range(k): #1. first window(K)
#     while dq and nums[dq[-1]]<=nums[i]: # we are maintaining decreasing ord in dq
#         dq.pop()                       # eg.[1,]->[3](1 is popped and 3 is added).   if dq back ele < curr ele pop dq back ele and append cur ele as we also dont need previous ele (short) for next window
#     dq.append(i)                       # eg.[3]->[3,-1] (back ele>cur, -1 we need for next window).   else add to dq   (when adding -> from back,  when deleting -> from front)        
# # remaining
# for i in range(k,n):
#     res.append(nums[dq[0]])  # max ele of window will be at front in dq
#     # 2. remove elements that are not in current window
#     while dq and dq[0]<i-k+1:   # we are storing indexes in dq instead of numbers because for this only, if dq ele index is less than window starting index remove.
#         dq.popleft() 
#     # 3. same process that give max ele of window at front
#     while dq and nums[dq[-1]]<=nums[i]:
#         dq.pop()
#     dq.append(i)
# res.append(nums[dq[0]]) # add last wind max ele that will be at front(only 1 max ele at last will be there)
# print(res)

#27, 493. Reverse Pairs ->(Count pairs (i, j) such that 0 ≤ i < j < n and nums[i] > 2 * nums[j])
# Input: nums = [1,3,2,3,1]
# Output: 2
# Explanation: The reverse pairs are:
# (1, 4) --> nums[1] = 3, nums[4] = 1, 3 > 2 * 1
# (3, 4) --> nums[3] = 3, nums[4] = 1, 3 > 2 * 1
# bruteforce tc=o(n2) sc=o(1)
# nums = [1,3,2,3,1]
# c=0
# n=len(nums)
# for i in range(n):
#     for j in range(i+1,n):
#         if nums[i]>nums[j]*2:
#             c+=1
# print(c)

# # optimal (merge sort) tc=o(2nlogn)->o(lognx(n+n)) sc=o(n) (for ct array)
# def reversePairs(nums):
#     def merge(l,m,h):
#         i=l
#         j=m+1
#         k=0
#         ct=[0]*(h-l+1)
#         while i<=m and j<=h:
#             if nums[i]<=nums[j]:
#                 ct[k]=nums[i]
#                 i+=1
#             else:
#                 ct[k]=nums[j]
#                 j+=1
#             k+=1
#         while i<=m:
#             ct[k]=nums[i]
#             i+=1
#             k+=1
#         while j<=h:
#             ct[k]=nums[j]
#             j+=1
#             k+=1
#         for i in range(len(ct)):
#             nums[l+i]=ct[i]
#     def countrp(l,m,h):
#         c=0
#         r=m+1
#         for i in range(l,m+1):
#             while r<=h and nums[i]>2*nums[r]:
#                 r+=1
#             c+=r-(m+1)
#         return c
#     def mergesort(l,h):
#         c=0
#         if l>=h:
#             return 0
#         m=(l+h)//2
#         c+=mergesort(l,m)
#         c+=mergesort(m+1,h)
#         c+=countrp(l,m,h)
#         merge(l,m,h)
#         return c
#     return mergesort(0,len(nums)-1)
# nums = [1,3,2,3,1]
# print(reversePairs(nums))

# 28. 84. Largest Rectangle in Histogram 
# Problem:
#  Given `heights[]`, return the **maximum rectangular area** in a histogram.
# Approach:
#  Use monotonic increasing stack** to find nearest smaller to left/right in O(n).
# Key Formula:
#  For each bar `i`:
# `area = heights[i] * (right[i] - left[i] - 1)`
#  Time: O(n)
#  Space: O(n)
# heights = [2,1,5,6,2,3]
# n=len(heights)
# ln=[0]*n
# rn=[0]*n
# stack=[]
# # right nearest shortest height for each height
# for i in range(n-1,-1,-1):
#     while stack and heights[stack[-1]]>=heights[i]:
#         stack.pop()
#     rn[i]= n if not stack else stack[-1]  # if no right nearest short height take n instead of -1 because eg. for 2 -> r-l-1 -> (-1)-1-1=-3 incorrect so, (6-1-1)=4 which gives correct . [2,1,5,6,2,3]
#     stack.append(i) 
# stack.clear()
# # left nearest shortest for each height
# for i in range(n):
#     while stack and heights[stack[-1]]>=heights[i]:
#         stack.pop()
#     ln[i]= -1 if not stack else stack[-1]
#     stack.append(i)
# maxarea=0 
# for i in range(n):
#     width=rn[i]-ln[i]-1
#     curarea=heights[i]*width
#     maxarea=max(maxarea,curarea)
# print(maxarea)