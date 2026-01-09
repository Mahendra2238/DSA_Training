# 1. 33. Search in Rotated Sorted Array
#  tc=o(logn) (modified binary search)
# nums = [4,5,6,7,0,1,2]
# target = 0
# l,h=0,len(nums)-1
# ans=-1
# while l<=h:
#     m=l+(h-l)//2
#     if nums[m]==target:
#         ans=m
#         break
#     if nums[l]<nums[m]: # left sorted  (check in sorted side)
#         if target>=nums[l] and target<=nums[m]: # if target is in left search in left else in right part
#             h=m-1
#         else:
#             l=m+1
#     else: # right sorted
#         if target>=nums[m] and target<=nums[h]:
#             l=m+1
#         else:
#             h=m-1
# print(ans)

# 2.852. Peak Index in a Mountain Array o(logn)
# arr = [0,2,1,0]
# l,h=1,len(arr)-2
# while l<=h:
#     m=(l+h)//2
#     if arr[m-1]<arr[m] and arr[m]>arr[m+1]:
#         print(m) 
#         break
#     if arr[m-1]<arr[m]: # seach right
#         l=m+1
#     else:
#         h=m-1 

# 3. 540. Single Element in a Sorted Array o(logn)
# nums = [3,3,7,7,10,11,11]
# def se(nums):
#     n=len(nums)
#     l,h=0,n-1
#     if n==1:
#         return nums[0]
#     while l<=h:
#         m=(l+h)//2
#         if m==0 and m+1<n and nums[1]!=nums[0]: # if ans is first ele
#             return nums[m]
#         if m==n-1 and m-1>=0 and nums[n-2]!=nums[n-1]: # if ans is last ele
#             return nums[m]
#         if (m > 0 and m < n - 1) and (nums[m-1]!=nums[m] and nums[m+1]!=nums[m]):
#             return nums[m]
#         if m%2==0: # index is even  # eg. m=9/2=4
#             if nums[m-1]==nums[m]:  # and when l and r of m len is even 
#                 h=m-1
#             else:
#                 l=m+1
#         else: #index  is odd       # eg. m=7/2=3
#             if nums[m-1]==nums[m]: # and when l and r of m len is odd
#                 l=m+1
#             else:
#                 h=m-1
# print(se(nums))

# # 4. Allocate Minimum Pages or Book Allocation (min pos of max pages)
# # Time: O(N * log(sum(arr) - max(arr)))    (N = number of books, binary search on pages)
# # Space: O(1) (constant space)
# def findPages(arr, k):
#     #code here
#     def isValid(maxAllocatedPages):
#         stu=1
#         pages=0
#         for i in range(len(arr)):
#             if arr[i]>maxAllocatedPages: #edgecase 
#                 return False
#             if pages+arr[i]<=maxAllocatedPages:
#                 pages+=arr[i]
#             else:
#                 stu+=1
#                 pages=arr[i]
#         return True if stu<=k else False
#     if k>len(arr):
#         return -1
#     # s=0
#     s=max(arr)
#     e=sum(arr)
#     ans=-1
#     while s<=e: # possible ans will be in this range (possible pages that can be allocated to k studeents)
#         m=s+(e-s)//2 #(Safe from integer overflow instead of (s+e)//2)
#         if isValid(m): # if pos check for min pos in left
#             ans=m
#             e=m-1
#         else:   # not valid (check right for pos value)
#             s=m+1
#     return ans
# arr=[2,1,3,4] # books where books[i]=pages
# k=2 # no of students
# print(findPages(arr,k))
            
# 5. painters partition o(nlog(sum)) (min pos of max time)
# def isPos(maxAllowedTime): # n
#     p=1
#     t=0
#     for i in range(len(boards)):
#         if t+boards[i]<=maxAllowedTime:
#            t+=boards[i]
#         else:
#            p+=1
#            t=boards[i]
#     return True if p<=n else False  
# boards=[40,30,10,20]
# n=2
# s=max(boards)
# e=sum(boards)
# ans=-1
# while s<=e:  # log(sum)
#     m=s+(e-s)//2
#     if isPos(m): # true->left (check for mintime pos than this)
#         ans=m
#         e=m-1
#     else:
#         s=m+1
# print(ans)

# 6. Aggressive cows (max pos of min dist) o(n*log(range))   
# def isPos(minAllowedDist):
#     cows=1 
#     lastStallPos=stalls[0] # cow1 pos at first
#     for i in range(n):
#         if stalls[i]-lastStallPos>=minAllowedDist: # if stalls dist(b/w last cow stallpos and this stall) is more than min dist allowed/req
#             cows+=1
#             lastStallPos=stalls[i]
#         if cows==c:
#             return True
#     return False
# stalls=[1,2,8,4,9]
# c=3
# stalls.sort()
# n=len(stalls)
# s=1 # min dist b/w 2 cows will be atleast 1
# e=stalls[n-1]-stalls[0]  # max dist - largest in arr - smallest in arr (9-1=8)
# ans=-1
# while s<=e:
#     m=s+(e-s)//2
#     if isPos(m): # right (find largest min dist possible)
#         ans=m
#         s=m+1
#     else:     # left (check smaller, dist is more not all cows can be placed) 
#         e=m-1
# print(ans)

# 7. 4. Median of Two Sorted Arrays o(log(n))
# def findMedianSortedArrays(nums1,nums2):
#     if len(nums1)>len(nums2): # for -> log(min(n1,n2)) (swap)
#         return findMedianSortedArrays(nums2,nums1)
#     n1,n2=len(nums1),len(nums2)
#     l=0
#     h=n1
#     while l<=h:
#         cut1=l+(h-l)//2
#         cut2=(n1+n2)//2-cut1 # total-cut1  so left and right part are equal (5 ele | 5 ele)
#         l1= float('-inf') if cut1==0 else nums1[cut1-1]   # if no ele inf, to compare [l1<r2 and l2<r1]
#         l2= float('-inf') if cut2==0 else nums2[cut2-1]
#         r1= float('inf') if cut1==n1 else nums1[cut1]
#         r2= float('inf') if cut2==n2 else nums2[cut2]
#         if l1>r2:
#             h=cut1-1
#         elif l2>r1:
#             l=cut1+1
#         else:   # if satisfied [l1<r2 and l2<r1]
#             if (n1+n2)%2==0:  #even len
#                 return (max(l1,l2)+min(r1,r2))/2
#             else:  #odd
#                 return min(r1,r2)
#     return 0
# nums1=[1,5,8,10,18,20]
# nums2=[2,3,6,7]
# print(findMedianSortedArrays(nums1,nums2))
