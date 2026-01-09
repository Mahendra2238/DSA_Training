# (1) 94. Binary Tree Inorder Traversal
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res=[]
        def inorder(node):
            if node:
                inorder(node.left)
                res.append(node.val)
                inorder(node.right)
        inorder(root)
        return res

# (2) 144. Binary Tree Preorder Traversal
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res=[]
        def preorder(root):
            if not root:
                return 
            res.append(root.val)
            preorder(root.left)
            preorder(root.right)
        preorder(root)
        return res

# (3) 145. Binary Tree Postorder Traversal
# time: O(n)
# space: O(h) h is the height of the tree
class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res=[]
        def postorder(root):
            if root:
                postorder(root.left)
                postorder(root.right)
                res.append(root.val)
        postorder(root)    
        return res

# (4) 101. Symmetric Tree
def isSymmetric(self, root: Optional[TreeNode]) -> bool:
    def mirror(a, b):
        if not a and not b: return True
        if not a or not b: return False
        return a.val == b.val and mirror(a.left, b.right) and mirror(a.right, b.left)
    return mirror(root, root)


# x = float('inf')      # positive infinity
# y = float('-inf')     # negative infinity
# or
# import math
# x = math.inf
# y = -math.inf

# (5) 783. Minimum Distance Between BST Nodes
class Solution:
    def minDiffInBST(self, root: Optional[TreeNode]) -> int:
        self.prev = None
        self.ans = float('inf')
        def inorder(node):
            if not node:
                return
            inorder(node.left)
            if self.prev is not None:
                self.ans = min(self.ans, node.val - self.prev)
            self.prev = node.val
            inorder(node.right)
        inorder(root)
        return self.ans
# (6) 100. Same Tree
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p or not q:
            return p == q
        isLeftSame = self.isSameTree(p.left, q.left)
        isRightSame = self.isSameTree(p.right, q.right)
        return isLeftSame and isRightSame and p.val == q.val
    
# (7) 94. Binary Tree Inorder Traversal (Morris Traversal)
    def inorderTraversalMorris(self, root: Optional[TreeNode]) -> List[int]:
        # Morris inorder traversal
        res = []
        curr = root
        while curr:
            # if left subtree is not there print curr and traverse right
            if curr.left is None:
                res.append(curr.val)
                curr = curr.right
            else:
                # find ip (right most node in left subtree)
                ip = curr.left # inorder predecessor
                while ip.right and ip.right != curr:
                    ip = ip.right
                if ip.right is None: # create thread
                    ip.right = curr
                    curr = curr.left
                else: # delete thread
                    ip.right = None
                    res.append(curr.val)
                    curr = curr.right
        return res

# (8) 543. Diameter of Binary Tree
# Time: O(n)
# Space: O(h) (recursion stack)lass Solution:
def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
   ans = 0  # stores the maximum diameter found
   def height(node):
       nonlocal ans
       if not node:
           return 0  # height of empty subtree
       lh = height(node.left)   # height of left subtree
       rh = height(node.right)  # height of right subtree
       ans = max(ans, lh + rh)  # update diameter at this node
       return max(lh, rh) + 1   # return height of current node
   height(root)  # start DFS
   return ans    # final diameter

# 9. (110) Balanced Binary Tree
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        def height(node):
            if not node:
                return 0  # height of empty subtree
            lh = height(node.left)   # left subtree height
            rh = height(node.right)  # right subtree height
            if lh == -1 or rh == -1:
                return -1  # already unbalanced
            
            if abs(lh - rh) > 1:
                return -1  # current node unbalanced
            return max(lh, rh) + 1  # return height
        
        return height(root) != -1  # true if balanced
    
# 10. (572) subtree of another tree
def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
    def isIdentical(p, q):
        if not p or not q:
            return p == q
        return p.val == q.val and isIdentical(p.left, q.left) and isIdentical(p.right, q.right)
    if root is None or subRoot is None:
        return subRoot == root
    if root.val == subRoot.val and isIdentical(root, subRoot):
        return True
    return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)

# 11. Top View of Binary Tre  gfg
from collections import deque
class Solution:
    def topView(self, root):
        if not root: return []
        q = deque()  # BFS queue
        m = {}  # hd -> first node value
        q.append((0, root))  # (horizontal distance, node)
        while q:
            hd, node = q.popleft()
            if hd not in m:
                m[hd] = node.data  # store topmost node
            if node.left:
                q.append((hd - 1, node.left))  # left child
            if node.right:
                q.append((hd + 1, node.right))  # right child
        return [m[k] for k in sorted(m)]  # left to right order

# bottom view
    def bottomView(self, root):
        # code here
        if not root: return []
        q = deque()  # BFS queue
        m = {}  # hd -> first node value
        q.append((0, root))  # (horizontal distance, node)
        while q:
            hd, node = q.popleft()
            m[hd] = node.data  # update bottom most
            if node.left:
                q.append((hd - 1, node.left))  # left child
            if node.right:
                q.append((hd + 1, node.right))  # right child
        return [m[k] for k in sorted(m)]  # left to right order
    
# (199) right view 
def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
    res = []
    def f(node, level):  # rt r l
        if not node: return 
        if level == len(res):
            res.append(node.val)
        f(node.right, level+1)  # (rt l r for left view )
        f(node.left, level+1)
    f(root, 0)
    return res
# left view
def leftView(self, root):
    # code here
    res = []
    if not root: return 
    q = deque()
    q.append(root)
    while q:
        n = len(q)
        if n>0:
           res.append(q[0].data)
        while n>0:
            node = q.popleft()
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
            n-=1
    return res
        
    
# 12. (513) Find bottomleft tree value
# Time Complexity: O(N) — every node is visited once.
# Space Complexity: O(N) — queue can hold up to all nodes in the last level (worst case).
from collections import deque
class Solution:
    def findBottomLeftValue(self, root: Optional[TreeNode]) -> int:
        if not root: return
        q = deque([root])  # BFS queue
        while q:
            node = q.popleft()
            bottomLeft = node.val  # last processed becomes bottom-left
            if node.right: q.append(node.right)  # push right first
            if node.left: q.append(node.left)  # then left
        return bottomLeft
        # if not root: return
        # q = deque()
        # q.append(root)

        # while q:
        #     n = len(q)
        #     while n>0:
        #         node = q.popleft()
        #         bottomLeft = node.val

        #         if node.right:
        #             q.append(node.right)
        #         if node.left:
        #             q.append(node.left)
        #         n-=1
        # return bottomLeft

# 102. Binary Tree Level Order Traversal
def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
    res = []
    if not root: return res
    q = deque([root])
    # q.append(root)
    while q:
        n = len(q)
        l = []
        while n>0:
            node = q.popleft()
            l.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
            n-=1
        res.append(l)
    return res

# 1161. Maximum Level Sum of a Binary Tree
def maxLevelSum(self, root: Optional[TreeNode]) -> int:
    if not root: return 0
    q = deque([root])
    level = 1
    bestLevel = 1
    maxSum = float('-inf')
    while q:
        n = len(q)
        currSum = 0
        while n > 0:
            node = q.popleft()
            currSum += node.val
            if node.left: q.append(node.left)
            if node.right: q.append(node.right)
            n -= 1
        if currSum > maxSum:
            maxSum = currSum
            bestLevel = level
        level += 1
    return bestLevel

    # res = []
    # if not root: return res
    # q = deque([root])
    # # q.append(root)
    # while q:
    #     n = len(q)
    #     l = []
    #     while n>0:
    #         node = q.popleft()
    #         l.append(node.val)
    #         if node.left:
    #             q.append(node.left)
    #         if node.right:
    #             q.append(node.right)
    #         n-=1
    #     res.append(l)
    # return res.index(max(res)) + 1

# 236. Lowest Common Ancestor of a Binary Tree
# Time: O(N)
# Space: O(H) recursion stack (O(N) worst case)
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root: return None  # base case
        if root == p or root == q: return root  # found p or q
        left = self.lowestCommonAncestor(root.left, p, q)   # search left
        right = self.lowestCommonAncestor(root.right, p, q) # search right
        if left and right: return root  # p & q in different subtrees
        return left if left else right  # propagate found node

# 1038. Binary Search Tree to Greater Sum Tree
# Time: O(N)
# Space: O(H) recursion stack
class Solution:
    def bstToGst(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        self.total = 0
        def dfs(node):
            if not node: return
            dfs(node.right)          # visit greater values first
            self.total += node.val
            node.val = self.total   # update node
            dfs(node.left)
        dfs(root)
        return root

# 105. Construct Binary Tree from Preorder and Inorder Traversal
# Time Complexity:
# O(n) — each node is processed once; inorder index lookup is O(1) via hashmap.
# Space Complexity:
# O(n) — hashmap + recursion stack (worst case: skewed tree).
class Solution:
    def buildTree(self, preorder, inorder):
        idx = {v: i for i, v in enumerate(inorder)}
        self.pre = 0

        def helper(l, r):
            if l > r:
                return None

            root_val = preorder[self.pre]
            self.pre += 1
            root = TreeNode(root_val)

            mid = idx[root_val]
            root.left = helper(l, mid - 1)
            root.right = helper(mid + 1, r)

            return root

        return helper(0, len(inorder) - 1)
        
        # self.preIdx = 0
        # def search(val, l, r):
        #     for i in range(l, r + 1):
        #         if inorder[i] == val:
        #             return i
        # def helper(l, r):
        #     if l > r: return None
        #     root_val = preorder[self.preIdx]
        #     self.preIdx += 1
        #     root = TreeNode(root_val)

        #     inIdx = search(root_val, l, r)
        #     root.left = helper(l, inIdx - 1)
        #     root.right = helper(inIdx + 1, r)
        #     return root
        # return helper(0, len(inorder) - 1)

# 106. Construct Binary Tree from Inorder and Postorder Traversal
class Solution:
    def buildTree(self, inorder, postorder):
        idx = {v: i for i, v in enumerate(inorder)}
        self.post = len(postorder) - 1

        def helper(l, r):
            if l > r:
                return None

            root_val = postorder[self.post]
            self.post -= 1
            root = TreeNode(root_val)

            mid = idx[root_val]
            root.right = helper(mid + 1, r)  
            #Postorder is Left → Right → Root.
            # When consuming it backwards, the order becomes:
            # Root → Right → Left
            root.left = helper(l, mid - 1)

            return root

        return helper(0, len(inorder) - 1)

# 114. Flatten Binary Tree to Linked List
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        self.NR = None

        def dfs(node): # reverse preorder (r l rt)
            if not node:
                return

            dfs(node.right)
            dfs(node.left)

            node.right = self.NR # NR - next right or last visited node
            node.left = None
            self.NR = node

        dfs(root)

# 662. Maximum Width of Binary Tree
from collections import deque

class Solution:
    def widthOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0

        q = deque([(root, 0)])   # (node, index)
        maxWid = 0

        while q:
            size = len(q)                # nodes in current level
            first = q[0][1]              # leftmost index
            last = q[-1][1]              # rightmost index
            maxWid = max(maxWid, last - first + 1)  # width of level

            for _ in range(size):
                node, idx = q.popleft()
                idx -= first             # normalize index

                if node.left:
                    q.append((node.left, 2*idx + 1))   # left child
                if node.right:
                    q.append((node.right, 2*idx + 2))  # right child

        return maxWid

# 103. Binary Tree Zigzag Level Order Traversal
from collections import deque
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        zigzag = []
        q = deque([root])
        flag = False  # False = left→right, True = right→left
        while q:
            size = len(q)
            level = []
            stack = []
            for _ in range(size):
                node = q.popleft()
                if flag:
                    stack.append(node.val)   # collect for reverse
                else:
                    level.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            while stack:
                level.append(stack.pop())    # reverse order
            zigzag.append(level)              # add level
            flag = not flag                   # toggle direction
        return zigzag

# 124. Binary Tree Maximum Path Sum
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.maxi = float('-inf')
        def mps(node):
            if not node:
                return 0
            # max path sum from left/right (ignore negatives)
            ls = max(0, mps(node.left))
            rs = max(0, mps(node.right))
            # update global maximum
            self.maxi = max(self.maxi, node.val + ls + rs)
            # return max downward path
            return node.val + max(ls, rs)
        mps(root)
        return self.maxi

# 24. 1483. kth ancestor of a tree node
class TreeAncestor:
    def __init__(self, n: int, parent: List[int]):
        LOG = 17  # max power needed (2^17 > 10^5)

        # up[i][v] = 2^i-th ancestor of node v
        self.up = [[-1] * n for _ in range(LOG)]

        # 2^0-th ancestor (direct parent)
        self.up[0] = parent

        # build ancestor table using DP
        for i in range(1, LOG):
            for v in range(n):
                p = self.up[i - 1][v]          # previous ancestor
                if p != -1:
                    self.up[i][v] = self.up[i - 1][p]

    def getKthAncestor(self, node: int, k: int) -> int:
        i = 0  # power index

        # move up according to binary representation of k
        while node != -1 and k:
            if k & 1:                           # if current bit is set
                node = self.up[i][node]         # jump 2^i ancestors
            k >>= 1                             # move to next bit
            i += 1                              # increase power
        return node                             # kth ancestor or -1
