# from collections import deque
# class Node:
#     def __init__(self, data):
#         self.data = data
#         self.left = None
#         self.right = None
# class AVL:
#     def __init__(self):
#         self.root = None
#     def height(self, root):
#         if root is None:
#             return -1
#         return max(self.height(root.left), self.height(root.right)) + 1
#     def balfac(self, root):
#         return self.height(root.left) - self.height(root.right)
#     def baltree(self, root):
#         if root is None:
#             return True
#         l = self.height(root.left)
#         r = self.height(root.right)
#         if abs(l - r) > 1:
#             return False
#         return self.baltree(root.left) and self.baltree(root.right)
#     def left_rot(self, x):
#         y = x.right
#         T2 = y.left
#         y.left = x
#         x.right = T2
#         return y
#     def right_rot(self, y):
#         x = y.left
#         T2 = x.right
#         x.right = y
#         y.left = T2
#         return x
#     def insert(self, root, x):
#         if not root:
#             return Node(x)
#         elif x < root.data:
#             root.left = self.insert(root.left, x)
#         elif x > root.data:
#             root.right = self.insert(root.right, x)
#         else:
#             return root 
#         balance = self.balfac(root)
#         if balance > 1 and x < root.left.data:
#             return self.right_rot(root)
#         if balance < -1 and x > root.right.data:
#             return self.left_rot(root)
#         if balance > 1 and x > root.left.data:
#             root.left = self.left_rot(root.left)
#             return self.right_rot(root)
#         if balance < -1 and x < root.right.data:
#             root.right = self.right_rot(root.right)
#             return self.left_rot(root)
#         return root
#     def insert_key(self, x):
#         self.root = self.insert(self.root, x)
#     def display(self):
#         if not self.root:
#             return
#         queue = deque([self.root])
#         while queue:
#             level_size = len(queue)
#             level_nodes = []
#             for _ in range(level_size):
#                 node = queue.popleft()
#                 if node:
#                     level_nodes.append(str(node.data))
#                     queue.append(node.left)
#                     queue.append(node.right)
#                 else:
#                     level_nodes.append(" ")
#             print(" ".join(level_nodes))
# tree = AVL()
# for val in [30, 20, 40, 10, 25, 50, 5]:
#     tree.insert_key(val)
# tree.display()  
    
# heaps
# - gets top priority value -> o(1)
# - insertion -> o(logn)
# - removal of top -> o(logn)

# complete binary tree -> A Complete Binary Tree is a special type of binary tree where:
# Every level is completely filled,
# Except possibly the last, which is filled from left to right.
# Key Properties:
# Height = ⌊log₂(n)⌋ where n is the number of nodes.
# Used in Heap implementations.
# Can be efficiently stored in arrays (index-based parent-child mapping).

# A **Heap** is a specialized **complete binary tree** used primarily for **priority-based operations**.
# ### Types:
# * **Max-Heap**: Parent ≥ children (root is the maximum).
# * **Min-Heap**: Parent ≤ children (root is the minimum).
# ### Properties:
# * Always a **complete binary tree**.
# * Implemented efficiently using **arrays** (no pointers needed).
# * Operations:
#   * **Insert**: O(log n) — bubble up
#   * **Delete root** (extract min/max): O(log n) — bubble down
#   * **Peek root**: O(1)
# ### Array Representation:
# For node at index `i`:
# * Left child → `2i + 1`
# * Right child → `2i + 2`
# * Parent → `(i - 1) // 2`
# ### Use Cases:
# * Priority Queues (e.g., task schedulers)
# * Dijkstra’s algorithm
# * Huffman encoding
# * Top-k problems (streaming data)
# ### Innovation Challenge:
# Reimagine heaps for:
# * **Streaming real-time data prioritization** (e.g., live alerts, sensor inputs).
# * **Edge AI** — heap-optimized scheduling for low-power neural modules.
# * **Blockchain gas optimization** — prioritize transactions efficiently.
# * **Multiplayer gaming** — dynamically rank players/events in real time.
# Think: Heap as the **brainstem of urgency** in data systems.

# DHCP (Dynamic Host Configuration Protocol) is the plug-and-play engine of networking — it automatically assigns IP addresses and other network settings (subnet mask, gateway, DNS) to devices joining a network.
# DNS (Domain Name System) is the Internet’s phonebook — it translates human-readable domain names (like openai.com) into IP addresses (like 104.22.1.46) that computers use to identify each other.

# Min heap
# class heapt:
#     def __init__(self):
#         self.heap=[]
#     def insert(self,data):
#         self.heap.append(data)
#         i=len(self.heap)-1
#         while i>0:
#             parent=(i-1)//2
#             if self.heap[parent]>self.heap[i]:
#                 self.heap[parent],self.heap[i]=self.heap[i],self.heap[parent]
#                 i=parent
#             else:
#                 break
#     def display(self):
#         print("Min heap: ",self.heap)
# h=heapt()
# arr=[20, 15, 30, 10, 25]
# for i in arr:
#     h.insert(i)
# h.display()

# class MaxHeap:
#     def __init__(self):
#         self.heap = []

#     def insert(self, val):
#         self.heap.append(val)
#         i = len(self.heap) - 1
#         while i > 0:
#             parent = (i - 1) // 2
#             if self.heap[i] > self.heap[parent]:
#                 self.heap[i], self.heap[parent] = self.heap[parent], self.heap[i]
#                 i = parent
#             else:
#                 break

#     def extract_max(self):
#         if not self.heap:
#             return None
#         if len(self.heap) == 1:
#             return self.heap.pop()
#         max_val = self.heap[0]
#         self.heap[0] = self.heap.pop()
#         self._heapify_down(0)
#         return max_val

#     def _heapify_down(self, i):
#         n = len(self.heap)
#         while True:
#             left = 2 * i + 1
#             right = 2 * i + 2
#             largest = i

#             if left < n and self.heap[left] > self.heap[largest]:
#                 largest = left
#             if right < n and self.heap[right] > self.heap[largest]:
#                 largest = right

#             if largest == i:
#                 break
#             self.heap[i], self.heap[largest] = self.heap[largest], self.heap[i]
#             i = largest

#     def display(self):
#         print("Max-Heap:", self.heap)

# max_heap = MaxHeap()

# for val in [10, 20, 5, 30, 25, 40, 50, 60]:
#     max_heap.insert(val)

# max_heap.display()

# print("Extracted Max:", max_heap.extract_max())

# max_heap.display()


