# HEAP:
# A heap is a complete binary tree with a heap property.

# COMPLETE BINARY TREE:
# All levels filled left → right.
# Last level may be incomplete.

# MIN HEAP:
# Parent ≤ Children.

# MAX HEAP:
# Parent ≥ Children.

# INDEX FORMULAS: (0-BASED)
# Parent = (i - 1) // 2
# Left Child = 2*i + 1
# Right Child = 2*i + 2

# HEAPIFY:
# Fixes heap from a node.
# Time Complexity: O(log n)

# INSERT:
# Add element at end.
# Bubble up to restore heap.
# Time Complexity: O(log n)

# DELETE:
# Replace with extreme value.
# Heapify to restore heap.
# Time Complexity: O(log n)

# BUILD HEAP:
# Bottom-up heapify.
# Time Complexity: O(n)

# HEAPSORT:
# Build heap and extract repeatedly.
# Time Complexity: O(n log n)
# Space Complexity: O(1) (in-place)

# APPLICATIONS:
# Priority Queue
# Scheduling
# Dijkstra Algorithm
# Top-K Problems

# -------- MAX HEAP IMPLEMENTATION --------

class MaxHeap:
    def __init__(self):
        self.h = []

    # index formulas
    def parent(self, i): return (i - 1) // 2
    def left(self, i): return 2 * i + 1
    def right(self, i): return 2 * i + 2

    # fix heap property from index i
    def heapify(self, i):
        n = len(self.h)
        largest = i
        l, r = self.left(i), self.right(i)

        if l < n and self.h[l] > self.h[largest]:
            largest = l
        if r < n and self.h[r] > self.h[largest]:
            largest = r

        if largest != i:
            self.h[i], self.h[largest] = self.h[largest], self.h[i]
            self.heapify(largest)

    # insert element
    def insert(self, key):
        self.h.append(key)
        i = len(self.h) - 1
        while i > 0 and self.h[self.parent(i)] < self.h[i]:
            self.h[i], self.h[self.parent(i)] = self.h[self.parent(i)], self.h[i]
            i = self.parent(i)

    # increase value at index i
    def increase_key(self, i, val):
        self.h[i] = val
        while i > 0 and self.h[self.parent(i)] < self.h[i]:
            self.h[i], self.h[self.parent(i)] = self.h[self.parent(i)], self.h[i]
            i = self.parent(i)

    # decrease value at index i
    def decrease_key(self, i, val):
        self.h[i] = val
        self.heapify(i)

    # remove max element
    def extract_max(self):
        if not self.h:
            return None
        root = self.h[0]
        self.h[0] = self.h[-1]
        self.h.pop()
        self.heapify(0)
        return root

    # delete element at index i
    def delete(self, i):
        self.increase_key(i, float('inf'))
        self.extract_max()

    # build heap from array
    def build_heap(self, arr):
        self.h = arr
        for i in range(len(self.h)//2 - 1, -1, -1):
            self.heapify(i)


# -------- HEAPSORT --------

def heapsort(arr):
    heap = MaxHeap()
    heap.build_heap(arr)
    res = []
    while heap.h:
        res.insert(0, heap.extract_max())
    return res


# -------- DRIVER CODE --------

heap = MaxHeap()
heap.insert(10)
heap.insert(30)
heap.insert(20)
heap.insert(5)

print("Heap:", heap.h)

heap.increase_key(2, 40)
print("After Increase Key:", heap.h)

heap.decrease_key(1, 15)
print("After Decrease Key:", heap.h)

heap.delete(1)
print("After Delete:", heap.h)

print("Extract Max:", heap.extract_max())
print("Final Heap:", heap.h)

arr = [4, 10, 3, 5, 1]
print("Heapsort:", heapsort(arr))

# -------- MERGE K SORTED ARRAYS --------Approach Used: Min Heap
# Heap Stores: (value, row_index, col_index)
# Why Heap: Always gives smallest among k arrays
# Steps:
# Push first element of each array
# Extract min and push next element from same array
# Time Complexity: O(N log k)
# Space Complexity: O(k)
# N = total elements, k = number of arrays

# 🔁 Alternate Approach (Theory)
# Merge arrays two at a time (Merge Sort idea)
# Time: O(N log k)
# Space: O(N)
import heapq
class Solution:
    def mergeArrays(self, mat):
        n=len(mat)          # number of arrays
        heap=[]             # min heap
        res=[]              # result
        for i in range(n):  # push first element of each array
            if mat[i]:
                heapq.heappush(heap,(mat[i][0],i,0))
        while heap:         # extract minimum
            val,ri,ci=heapq.heappop(heap)
            res.append(val)
            if ci+1<len(mat[ri]):  # push next element
                heapq.heappush(heap,(mat[ri][ci+1],ri,ci+1))

        return res
# -------- DRIVER CODE --------
mat=[[1,4,7],[2,5,8],[3,6,9]]
obj=Solution()
print(obj.mergeArrays(mat))
# def merge(a,b):           # merge two arrays approach
        #     i=j=0
        #     res=[]
        #     while i<len(a) and j<len(b):
        #         if a[i]<=b[j]:
        #             res.append(a[i]); i+=1
        #         else:
        #             res.append(b[j]); j+=1
        #     res.extend(a[i:])     # remaining elements
        #     res.extend(b[j:])
        #     return res
        # def divide(arr):          # divide arrays
        #     if len(arr)==1:
        #         return arr[0]
        #     mid=len(arr)//2
        #     left=divide(arr[:mid])
        #     right=divide(arr[mid:])
        #     return merge(left,right)
        # return divide(mat)

# -------- 347. TOP K FREQUENT ELEMENTS --------
from collections import Counter
import heapq
from typing import List

class Solution:
    # ---------- Max Heap Approach ----------
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        freq = Counter(nums)                        # count frequency
        heap = [(-count, num) for num, count in freq.items()]  # max-heap using -count
        heapq.heapify(heap)                         # build heap
        res = []
        for _ in range(k):
            res.append(heapq.heappop(heap)[1])     # extract top k numbers
        return res
nums = [1,1,1,2,2,3]
k = 2
obj = Solution()
print(obj.topKFrequent(nums, k))
    # # ---------- Min Heap Approach ----------
    # def topKFrequent(self, nums: List[int], k: int) -> List[int]:
    #     freq = Counter(nums)
    #     heap = []
    #     for num, count in freq.items():
    #         heapq.heappush(heap, (count, num))     # min-heap
    #         if len(heap) > k:                       # keep size k
    #             heapq.heappop(heap)
    #     return [num for count, num in heap]        # extract numbers

    # # ---------- Bucket Sort Approach ----------
    # def topKFrequent(self, nums: List[int], k: int) -> List[int]:
    #     freq = Counter(nums)
    #     n = len(nums)
    #     bucket = [[] for _ in range(n+1)]          # bucket index = frequency
    #     for num, count in freq.items():
    #         bucket[count].append(num)
    #     res = []
    #     for i in range(n, 0, -1):                  # iterate from high frequency
    #         for num in bucket[i]:
    #             res.append(num)
    #             if len(res) == k:
    #                 return res

# -------- 912. SORT AN ARRAY --------
class Solution:
    def sortArray(self, nums: List[int]) -> List[int]:
        heapq.heapify(nums)
        res=[]
        while nums:
            res.append(heapq.heappop(nums))
        return res
        # def merge(nums,s,m,e):
        #     n1=m-s+1
        #     n2=e-m  # fe-(m+1)+1 = e-m-1+1 = e-m
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
        # return nums
    
# -------- 378. KTH SMALLEST ELEMENT IN A SORTED MATRIX --------
import heapq
from typing import List
class Solution:
    # -------- Min Heap Approach --------
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        n = len(matrix)
        heap = []
        for r in range(n):                          # push first element of each row
            heapq.heappush(heap,(matrix[r][0],r,0))
        for _ in range(k):                          # pop k times
            val,r,c = heapq.heappop(heap)
            if c+1 < n:                             # push next element from same row
                heapq.heappush(heap,(matrix[r][c+1],r,c+1))
        return val                                  # kth smallest

    # -------- Binary Search Approach --------
    # def kthSmallest_BinarySearch(self, matrix: List[List[int]], k: int) -> int:
    #     n = len(matrix)
    #     low = matrix[0][0]                          # minimum value
    #     high = matrix[-1][-1]                       # maximum value
    #     while low < high:                           # binary search on values
    #         mid = (low + high) // 2
    #         count = 0
    #         c = n - 1                               # last column
    #         for r in range(n):                      # count <= mid
    #             while c >= 0 and matrix[r][c] > mid:
    #                 c -= 1
    #             count += (c + 1)
    #         if count < k:                           # kth element is larger
    #             low = mid + 1
    #         else:
    #             high = mid
    #     return low                                  # kth smallest value
# -------- DRIVER CODE --------
matrix = [[1,5,9],[10,11,13],[12,13,15]]
k = 8
obj = Solution()
print("Heap:", obj.kthSmallest(matrix,k))
# print("Binary Search:", obj.kthSmallest_BinarySearch(matrix,k))

# -------- 295. FIND MEDIAN FROM DATA STREAM --------
import heapq
class MedianFinder:
    def __init__(self):
        self.left_max_heap = []      # max heap (lower half, stored as negative)
        self.right_min_heap = []     # min heap (upper half)

    def addNum(self, num: int) -> None:
        # decide heap
        if not self.left_max_heap or num <= -self.left_max_heap[0]:
            heapq.heappush(self.left_max_heap, -num)     # push to left
        else:
            heapq.heappush(self.right_min_heap, num)     # push to right

        # balance sizes (# always maintain left max heap size one greater than right min heap size or equal)
        if len(self.left_max_heap) > len(self.right_min_heap) + 1:
            heapq.heappush(self.right_min_heap, -heapq.heappop(self.left_max_heap))
        elif len(self.left_max_heap) < len(self.right_min_heap):
            heapq.heappush(self.left_max_heap, -heapq.heappop(self.right_min_heap))

    def findMedian(self) -> float:
        # even count
        if len(self.left_max_heap) == len(self.right_min_heap):
            return (-self.left_max_heap[0] + self.right_min_heap[0]) / 2
        # odd count
        return -self.left_max_heap[0]

# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()

# 632. Smallest Range Covering Elements from K Lists
# (Min-Heap (K-way merge) Sliding Window)
import heapq
from typing import List
class Solution:
    def smallestRange(self, nums: List[List[int]]) -> List[int]:
        heap = []                              # min-heap: (value, listIdx, elemIdx)
        currMax = float('-inf')                # max among current window
        res = [float('-inf'), float('inf')]    # best (smallest) range
        for i in range(len(nums)):
            val = nums[i][0]
            heapq.heappush(heap, (val, i, 0))  # push first element of each list
            currMax = max(currMax, val)        # track max
        while True:
            currMin, li, ei = heapq.heappop(heap)  # smallest element
            if currMax - currMin < res[1] - res[0]:
                res = [currMin, currMax]       # update answer
            if ei + 1 == len(nums[li]):
                break                          # one list exhausted
            nextVal = nums[li][ei + 1]
            heapq.heappush(heap, (nextVal, li, ei + 1))  # push next element
            currMax = max(currMax, nextVal)    # update max
        return res
# -------- DRIVER CODE --------
nums = [[4,10,15,24,26],[0,9,12,20],[5,18,22,30]]
obj = Solution()
print(obj.smallestRange(nums))
        # k = len(nums)
        # karr = [0] * k              # pointers for each list
        # resRange = [float('-inf'), float('inf')]  # best range

        # while True:
        #     minEl = float('inf')
        #     maxEl = float('-inf')
        #     minElIdx = 0

        #     for i in range(k):
        #         el = nums[i][karr[i]]   # current element
        #         if el < minEl:
        #             minEl = el          # smallest value
        #             minElIdx = i        # list of smallest
        #         maxEl = max(maxEl, el)  # largest value

        #     if maxEl - minEl < resRange[1] - resRange[0]:
        #         resRange = [minEl, maxEl]  # update answer

        #     karr[minElIdx] += 1         # move min list forward
        #     if karr[minElIdx] == len(nums[minElIdx]):
        #         break                   # one list exhausted

        # return resRange


