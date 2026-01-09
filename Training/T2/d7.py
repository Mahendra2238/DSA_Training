# prims 
# 1. Adjacency List (Dictionary) – Best for flexibility
# import heapq
# def prims_algorithm(graph, start):
#     n = len(graph)
#     visited = [False] * n
#     min_heap = [(0, start)]  # (cost, node)
#     total_cost = 0
#     while min_heap:
#         cost, u = heapq.heappop(min_heap)
#         if visited[u]: continue
#         visited[u] = True
#         total_cost += cost
#         for v, weight in graph[u]:
#             if not visited[v]:
#                 heapq.heappush(min_heap, (weight, v))
#     return total_cost
# graph = {
#     0: [(1, 2), (3, 6)],
#     1: [(0, 2), (2, 3), (3, 8), (4, 5)],
#     2: [(1, 3), (4, 7)],
#     3: [(0, 6), (1, 8), (4, 9)],
#     4: [(1, 5), (2, 7), (3, 9)]
# }
# total_cost = prims_algorithm(graph, start=0)
# print("Minimum Spanning Tree cost:", total_cost)

#  2. Adjacency Matrix – Compact, great for dense graphs
# def matrix_to_adj_list(matrix):
#     graph = {i: [] for i in range(len(matrix))}
#     for i in range(len(matrix)):
#         for j in range(len(matrix)):
#             if matrix[i][j] != 0:
#                 graph[i].append((j, matrix[i][j]))
#     return graph
# matrix = [
#     [0, 2, 0, 6, 0],
#     [2, 0, 3, 8, 5],
#     [0, 3, 0, 0, 7],
#     [6, 8, 0, 0, 9],
#     [0, 5, 7, 9, 0]
# ]
# graph = matrix_to_adj_list(matrix)
# total_cost = prims_algorithm(graph, start=0)
# print("Minimum Spanning Tree cost:", total_cost)

# class DisjointSet:
#     def __init__(self, n):
#         self.parent = list(range(n))
#     def find(self, u):
#         if u != self.parent[u]:
#             self.parent[u] = self.find(self.parent[u])
#         return self.parent[u]
#     def union(self, u, v):
#         pu, pv = self.find(u), self.find(v)
#         if pu == pv:
#             return False
#         self.parent[pu] = pv
#         return True
# def kruskal(n, edges):
#     ds = DisjointSet(n)
#     mst_weight = 0
#     edges.sort(key=lambda x: x[2])  # Sort by weight
#     for u, v, w in edges:
#         if ds.union(u, v):
#             mst_weight += w
#     return mst_weight
# edges = [
#     (0, 1, 2),
#     (0, 3, 6),
#     (1, 2, 3),
#     (1, 3, 8),
#     (1, 4, 5),
#     (2, 4, 7),
#     (3, 4, 9)
# ]
# n = 5  # number of nodes
# print("MST cost:", kruskal(n, edges))

# from collections import defaultdict
# class UndirectedGraph:
#     def __init__(self, vertices):
#         self.V = vertices
#         self.graph = defaultdict(list)
#     def add_edge(self, u, v):
#         self.graph[u].append(v)
#         self.graph[v].append(u)
#     def is_cyclic_util(self, v, visited, parent):
#         visited[v] = True
#         for neighbor in self.graph[v]:
#             if not visited[neighbor]:
#                 if self.is_cyclic_util(neighbor, visited, v):
#                     return True
#             elif parent != neighbor:
#                 return True
#         return False
#     def is_cyclic(self):
#         visited = [False] * self.V
#         for node in range(self.V):
#             if not visited[node]:
#                 if self.is_cyclic_util(node, visited, -1):
#                     return True
#         return False
# g = UndirectedGraph(5)
# g.add_edge(0, 1)
# g.add_edge(1, 2)
# g.add_edge(2, 3)
# g.add_edge(3, 0)  # introduces a cycle
# print("Cyclic" if g.is_cyclic() else "Acyclic")

# from collections import deque, defaultdict
# class Graph:
#     def __init__(self, vertices):
#         self.V = vertices
#         self.graph = defaultdict(list)
#     def add_edge(self, u, v):
#         self.graph[u].append(v)
#         self.graph[v].append(u)  # Remove if directed
#     def shortest_distance(self, src, dest):
#         visited = [False] * self.V
#         dist = [0] * self.V
#         queue = deque([src])
#         visited[src] = True
#         while queue:
#             node = queue.popleft()
#             for neighbor in self.graph[node]:
#                 if not visited[neighbor]:
#                     dist[neighbor] = dist[node] + 1
#                     queue.append(neighbor)
#                     visited[neighbor] = True
#                     if neighbor == dest:
#                         return dist[neighbor]
#         return -1  # unreachable
# g = Graph(6)
# g.add_edge(0, 1)
# g.add_edge(1, 2)
# g.add_edge(2, 5)
# g.add_edge(0, 3)
# g.add_edge(3, 4)
# g.add_edge(4, 5)
# print("Distance:", g.shortest_distance(0, 5))  # Output: 3

# def findWords(board,words):
#     m=len(board)
#     n=len(board[0])
#     res=[]
#     def find(word,i,j,l):
#         if l==len(word):
#             return True
#         if i<0 or i>=m or j<0 or j>=n or board[i][j]!=word[l]:
#             return False
#         tmp=board[i][j]
#         board[i][j]='$'
#         found=find(word,i-1,j,l+1) or find(word,i+1,j,l+1) or find(word,i,j+1,l+1) or find(word,i,j-1,l+1)                
#         board[i][j]=tmp
#         return found
#     def search(word):
#         for i in range(m):
#             for j in range(n):
#                 if board[i][j]==word[0] and find(word,i,j,0):
#                     return True
#         return False
#     for word in words:
#         if search(word):
#             res.append(word)
#     return res                
# # board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]]
words = ["oath","pea","eat","rain"]
# board = [["a","b"],["c","d"]]
# words = ["abcb"]
# print(findWords(board,words))

trie = {}
for word in words:
    node = trie
    for c in word:
        node = node.setdefault(c, {})
        print(node)
    node['#'] = word 
    # print(trie)