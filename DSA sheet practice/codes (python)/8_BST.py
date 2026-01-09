# ================================
# BST & HEAP PRACTICE – LEETCODE
# ================================


# ------------------------------------------------
# 1. 703. Kth Largest Element in a Stream
# ------------------------------------------------
import heapq
from typing import List

class KthLargest:
    def __init__(self, k: int, nums: List[int]):
        self.k = k                              # store k
        self.minHeap = []                       # min heap

        for num in nums:
            heapq.heappush(self.minHeap, num)  # add number
            if len(self.minHeap) > k:           # keep only k elements
                heapq.heappop(self.minHeap)

    def add(self, val: int) -> int:
        heapq.heappush(self.minHeap, val)       # add new value
        if len(self.minHeap) > self.k:
            heapq.heappop(self.minHeap)         # remove smallest
        return self.minHeap[0]                  # kth largest


# ------------------------------------------------
# 2. 108. Convert Sorted Array to Binary Search Tree
# Time: O(n) | Space: O(log n)
# ------------------------------------------------
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        def helper(s, e):
            if s > e:                      # base case
                return None
            mid = s + (e - s) // 2         # middle index
            root = TreeNode(nums[mid])     # create node
            root.left = helper(s, mid - 1) # left subtree
            root.right = helper(mid + 1, e)# right subtree
            return root
        return helper(0, len(nums) - 1)


# ------------------------------------------------
# 3. 98. Validate Binary Search Tree
# Time: O(n) | Space: O(h)
# ------------------------------------------------
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def helper(node, minv, maxv):
            if not node:
                return True
            if minv is not None and node.val <= minv:
                return False
            if maxv is not None and node.val >= maxv:
                return False
            return helper(node.left, minv, node.val) and helper(node.right, node.val, maxv)
        return helper(root, None, None)


# ------------------------------------------------
# 4. 230. Kth Smallest Element in a BST
# ------------------------------------------------
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        self.prevOrder = 0
        def helper(root):
            if not root: return -1
            if root.left:
                leftAns = helper(root.left)
                if leftAns != -1:
                    return leftAns
            self.prevOrder += 1
            if self.prevOrder == k:
                return root.val
            return helper(root.right)
        return helper(root)


# ------------------------------------------------
# 5. 235. Lowest Common Ancestor of a BST
# Time: O(h) | Space: O(h)
# ------------------------------------------------
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root:
            return None
        if p.val < root.val and q.val < root.val:
            return self.lowestCommonAncestor(root.left, p, q)
        elif p.val > root.val and q.val > root.val:
            return self.lowestCommonAncestor(root.right, p, q)
        else:
            return root


# ------------------------------------------------
# 6. 116. Populating Next Right Pointers in Each Node
# Time: O(n) | Space: O(n)
# ------------------------------------------------
from collections import deque

class Solution:
    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        if root is None or root.left is None: return root
        q = deque([root, None])
        prev = None
        while q:
            curr = q.popleft()
            if curr is None:
                prev = None
                if q: q.append(None)
            else:
                if prev: prev.next = curr
                if curr.left: q.append(curr.left)
                if curr.right: q.append(curr.right)
                prev = curr
        return root


# ------------------------------------------------
# 7. 99. Recover Binary Search Tree
# Time: O(n) | Space: O(h)
# ------------------------------------------------
class Solution:
    def recoverTree(self, root: Optional[TreeNode]) -> None:
        self.prev = None
        self.first = None
        self.sec = None   # if not self., use nonlocal and declare again in func 

        def inorder(node):
            if not node: return
            inorder(node.left)
            if self.prev and self.prev.val > node.val:
                if not self.first: self.first = self.prev
                self.sec = node
            self.prev = node
            inorder(node.right)

        inorder(root)
        if self.first and self.sec:
            self.first.val, self.sec.val = self.sec.val, self.first.val


# ------------------------------------------------
# 7(b). 99. Recover Binary Search Tree (Morris)
# Time: O(n) | Space: O(1)
# ------------------------------------------------
# Morris inorder traversal (iterative, no stack)
self.prev = None
self.f = None
self.s = None

while root:
    if not root.left:
        if self.prev and self.prev.val > root.val:
            if not self.f: self.f = self.prev
            self.s = root
        self.prev = root
        root = root.right
    else:
        ip = root.left
        while ip.right and ip.right != root:
            ip = ip.right
        if not ip.right:
            ip.right = root
            root = root.left
        else:
            ip.right = None
            if self.prev and self.prev.val > root.val:
                if not self.f: self.f = self.prev
                self.s = root
            self.prev = root
            root = root.right

if self.f and self.s:
    self.f.val, self.s.val = self.s.val, self.f.val

# ============================================================
# 8. 1008. Construct Binary Search Tree from Preorder Traversal
# ============================================================

# Time: O(n) | Space: O(n)  # recursion stack
import math
from typing import List, Optional

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def bstFromPreorder(self, preorder: List[int]) -> Optional[TreeNode]:
        self.i = 0  # shared index in preorder

        def buildBST(bound):
            if self.i == len(preorder) or preorder[self.i] > bound:  # stop condition
                return None
            root = TreeNode(preorder[self.i])  # create node
            self.i += 1
            root.left = buildBST(root.val)     # left subtree < root
            root.right = buildBST(bound)       # right subtree < bound
            return root

        return buildBST(math.inf)  # initial bound = +inf


# ============================================================
# 9. 173. Binary Search Tree Iterator
# ============================================================

# Time: O(1) amortized per operation | Space: O(h)
from typing import Optional

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class BSTIterator:

    def __init__(self, root: Optional[TreeNode]):
        self.stack = []                 # stack to simulate inorder
        self.storeLeftNodes(root)       # push all left nodes

    def next(self) -> int:
        node = self.stack.pop()         # next smallest node
        if node.right:
            self.storeLeftNodes(node.right)  # process right subtree
        return node.val                 # return value

    def hasNext(self) -> bool:
        return len(self.stack) > 0      # check remaining nodes

    def storeLeftNodes(self, root):
        while root:
            self.stack.append(root)     # push node
            root = root.left            # go left

# ============================================================
# 10. (GFG)Flatten BST to Sorted Linked List (Corrected)
# ============================================================
# Time: O(n) | Space: O(h)
'''
class Node:
    def __init__(self, val):
        self.data = val
        self.left = None
        self.right = None
'''
class Solution:
    def flattenBST(self, root):
        prev = None      # previous inorder node
        head = None      # head of flattened list

        def inorder(node):
            nonlocal prev, head
            if not node: return
            inorder(node.left)           # visit left
            if not prev:
                head = node              # first node becomes head
            else:
                prev.right = node        # link prev → curr
                prev.left = None         # remove left pointer
            prev = node                  # update prev
            inorder(node.right)          # visit right

        inorder(root)
        if prev: prev.left = None        # ensure last node left is None
        return head                      # return flattened list head

# ============================================================
#  11. gfg Predecessor and Successor in BST
# ============================================================
# Time: O(h) | Space: O(1)
'''
class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
'''
class Solution:
    def findPreSuc(self, root, key):
        pred = None  # predecessor
        succ = None  # successor
        curr = root

        while curr:
            if key < curr.data:
                succ = curr              # possible successor
                curr = curr.left
            elif key > curr.data:
                pred = curr              # possible predecessor
                curr = curr.right
            else:
                # predecessor = rightmost of left subtree
                if curr.left:
                    temp = curr.left
                    while temp.right:
                        temp = temp.right
                    pred = temp

                # successor = leftmost of right subtree
                if curr.right:
                    temp = curr.right
                    while temp.left:
                        temp = temp.left
                    succ = temp
                break

        return [pred, succ]

# ============================================================
# 13. gfg Largest BST in a Binary Tree
# ============================================================

# Time: O(n) | Space: O(h)

import math
import sys
sys.setrecursionlimit(10**7)

class info:
    def __init__(self, mn, mx, sz):
        self.mn = mn
        self.mx = mx
        self.sz = sz

class Solution:
    def largestBst(self, root):
        def helper(node):
            if node is None:
                return info(math.inf, -math.inf, 0)

            left = helper(node.left)
            right = helper(node.right)

            if node.data > left.mx and node.data < right.mn:
                return info(
                    min(node.data, left.mn),
                    max(node.data, right.mx),
                    left.sz + right.sz + 1
                )
            else:
                return info(-math.inf, math.inf, max(left.sz, right.sz))

        return helper(root).sz

# ============================================================
# 14. 297. Serialize and Deserialize Binary Tree
# ============================================================
# Time: O(n) | Space: O(n)
from collections import deque
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Codec:
    def serialize(self, root):
        if not root: return ""
        q = deque([root])          # BFS queue
        res = []
        while q:
            node = q.popleft()
            if not node:
                res.append("n")    # null marker
            else:
                res.append(str(node.val))
                q.append(node.left)
                q.append(node.right)
        return ",".join(res)
    def deserialize(self, data):
        if not data: return None
        values = data.split(",")
        root = TreeNode(int(values[0]))
        q = deque([root])
        i = 1
        while q:
            node = q.popleft()
            # left child
            if values[i] != "n":
                node.left = TreeNode(int(values[i]))
                q.append(node.left)
            i += 1
            # right child
            if values[i] != "n":
                node.right = TreeNode(int(values[i]))
                q.append(node.right)
            i += 1
        return root
    
# Merge two BSTs and return sorted inorder traversal
# Time: O(n + m)
# Space: O(n + m)

'''
class Node:
    def __init__(self, val):
        self.data = val
        self.left = None
        self.right = None
'''

class Solution:
    def merge(self, root1, root2):

        # Inorder traversal to get sorted array
        def inorder(root, arr):
            if not root: return
            inorder(root.left, arr)
            arr.append(root.data)
            inorder(root.right, arr)

        arr1, arr2 = [], []
        inorder(root1, arr1)        # inorder of first BST
        inorder(root2, arr2)        # inorder of second BST

        # Merge two sorted arrays
        merged = []
        i = j = 0
        while i < len(arr1) and j < len(arr2):
            if arr1[i] <= arr2[j]:
                merged.append(arr1[i]); i += 1
            else:
                merged.append(arr2[j]); j += 1
        merged.extend(arr1[i:])
        merged.extend(arr2[j:])

        # Build balanced BST from sorted array
        def sortedArrayToBST(arr, s, e):
            if s > e: return None
            mid = (s + e) // 2
            root = Node(arr[mid])
            root.left = sortedArrayToBST(arr, s, mid - 1)
            root.right = sortedArrayToBST(arr, mid + 1, e)
            return root

        new_root = sortedArrayToBST(merged, 0, len(merged) - 1)

        # Final inorder traversal (answer)
        result = []
        def inorder_result(root):
            if not root: return
            inorder_result(root.left)
            result.append(root.data)
            inorder_result(root.right)

        inorder_result(new_root)
        return result


