# 1. Leetcode - 733.Flood Fill
class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        iniCol = image[sr][sc]                 # starting pixel color
        if iniCol == color:                    # no change needed
            return image
        n, m = len(image), len(image[0])       # grid size
        dr = [-1, 0, 1, 0]                     # row directions
        dc = [0, 1, 0, -1]                     # column directions
        def dfs(r, c):
            image[r][c] = color                # fill current pixel
            for i in range(4):
                nr, nc = r + dr[i], c + dc[i]  # neighbor cell
                if 0 <= nr < n and 0 <= nc < m and image[nr][nc] == iniCol:
                    dfs(nr, nc)                # recursive DFS
        dfs(sr, sc)                            # start DFS
        return image                           # return updated image
# Time Complexity: O(n * m)
# Space Complexity: O(n * m)

# 2. 133. Clone Graph (DFS)
from typing import Optional

class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        if not node:                           # empty graph
            return None
        hm = {}                                # original -> cloned mapping
        def clone(curr):
            if curr in hm:                     # already cloned
                return hm[curr]
            newNode = Node(curr.val)           # create clone
            hm[curr] = newNode
            for nei in curr.neighbors:         # process neighbors
                newNode.neighbors.append(clone(nei))
            return newNode
        return clone(node)                     # start DFS
# Time Complexity: O(V + E)
# Space Complexity: O(V)


# 3. LeetCode - 684. Redundant Connection (DSU)
class DSU:
    def __init__(self, n):
        self.parent = [i for i in range(n + 1)]   # parent initialization
        self.rank = [0] * (n + 1)                 # rank array

class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        n = len(edges)
        dsu = DSU(n)                              # DSU object
        parent = dsu.parent
        rank = dsu.rank

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])      # path compression
            return parent[x]

        def union(x, y):
            x_parent = find(x)
            y_parent = find(y)
            if x_parent == y_parent:              # cycle detected
                return False
            if rank[x_parent] > rank[y_parent]:
                parent[y_parent] = x_parent
            elif rank[y_parent] > rank[x_parent]:
                parent[x_parent] = y_parent
            else:
                parent[y_parent] = x_parent
                rank[x_parent] += 1
            return True

        for u, v in edges:
            if not union(u, v):                   # redundant edge
                return [u, v]
        return []
# Time Complexity: O(E * α(N))
# Space Complexity: O(N)


# DFS
#         def dfs(u, target, vis):
#             vis[u] = True
#             if u == target:                    # path found
#                 return True
#             for nei in adj[u]:
#                 if not vis[nei]:
#                     if dfs(nei, target, vis):
#                         return True
#             return False

#         n = len(edges)
#         adj = {i: [] for i in range(1, n + 1)} # adjacency list
#         for u, v in edges:
#             vis = [False] * (n + 1)            # visited array
#             if dfs(u, v, vis):# (and u in adj and v in adj) # cycle check before adding edge
#                 return [u, v]
#             adj[u].append(v)                   # add edge
#             adj[v].append(u)
#         return []
# # Time Complexity: O(E * (V + E))
# # Space Complexity: O(V + E)

# ============================================================================
# practice
# ===============================
# 1. Print All Paths from Source to Destination (Directed Graph)
# ===============================
def print_all_paths(graph, src, dest):
    def dfs(curr, path, vis):
        path.append(curr)                 # add current node to current path
        vis[curr] = True                   # mark as visited

        if curr == dest:                   # destination reached
            print(" -> ".join(map(str, path)))  # print current path
        else:
            for nei in graph.get(curr, []):     # explore neighbors
                if not vis[nei]:
                    dfs(nei, path, vis)        # recursive DFS

        path.pop()                         # backtrack: remove current node
        vis[curr] = False                  # unmark for other paths

    V = max(graph.keys()) + 1             # number of nodes
    vis = [False] * V                     # visited array
    dfs(src, [], vis)                     # start DFS from source


# Example usage
graph = {
    0: [1, 2],
    1: [2, 3],
    2: [3],
    3: []
}

print("All paths from 0 to 3:")
print_all_paths(graph, 0, 3)


# ===============================
# 2. Undirected Graph Cycle Detection (DFS)
# ===============================
class UndirectedGraphCycle:
    def isCyclic(self, V, edges):
        # build adjacency list
        adj = {i: [] for i in range(V)}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)                     # undirected graph

        vis = [False] * V

        def dfs(curr, parent):
            vis[curr] = True
            for nei in adj[curr]:
                if not vis[nei]:
                    if dfs(nei, curr):
                        return True
                elif nei != parent:             # visited and not parent -> cycle
                    return True
            return False

        for i in range(V):
            if not vis[i]:
                if dfs(i, -1):
                    return True
        return False
# Time Complexity: O(V + E)
# Space Complexity: O(V)


# ===============================
# 3. Directed Graph Cycle Detection (DFS)
# ===============================
class DirectedGraphCycle:
    def isCyclic(self, V, edges):
        # build adjacency list
        adj = {i: [] for i in range(V)}
        for u, v in edges:
            adj[u].append(v)

        vis = [False] * V                     # visited array
        rec = [False] * V                     # recursion stack

        def dfs(curr):
            vis[curr] = True                  # mark visited
            rec[curr] = True                  # mark in current path
            for nei in adj[curr]:
                if rec[nei]:                  # back-edge found
                    return True
                if not vis[nei]:
                    if dfs(nei):
                        return True
            rec[curr] = False                 # remove from path
            return False

        for i in range(V):
            if not vis[i]:
                if dfs(i):                    # cycle check
                    return True
        return False
# Time Complexity: O(V + E)
# Space Complexity: O(V)
# ==============================================================

# 4. LeetCode 310. Minimum Height Trees (BFS / Leaf Trimming)
from collections import deque
from typing import List
class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        if n == 1: return [0]                    # single node
        adj = {i: [] for i in range(n)}          # adjacency list
        indegree = [0] * n                       # degree array
        for u, v in edges:
            adj[u].append(v); adj[v].append(u)   # undirected edge
            indegree[u] += 1; indegree[v] += 1   # update degrees
        q = deque()
        for i in range(n):
            if indegree[i] == 1: q.append(i)     # initial leaves
        while n > 2:                             # trim leaves
            size = len(q); n -= size
            for _ in range(size):
                u = q.popleft()
                for v in adj[u]:
                    indegree[v] -= 1
                    if indegree[v] == 1: q.append(v)
        return list(q)                           # MHT roots
# Time Complexity: O(N)
# Space Complexity: O(N)

# 5. # LeetCode 1584. Min Cost to Connect All Points (Prim’s Algorithm)
import heapq
from typing import List
class Solution:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        n = len(points)                          # number of points
        visited = set([0])                       # start from point 0
        heap = []                                # min-heap (cost, point)
        cost = 0                                 # total MST cost
        x0, y0 = points[0]
        for i in range(1, n):                    # initial edges from point 0
            xi, yi = points[i]
            d = abs(x0 - xi) + abs(y0 - yi)
            heapq.heappush(heap, (d, i))
        while len(visited) < n:                  # until all points included
            w, v = heapq.heappop(heap)
            if v in visited:                     # skip if already added
                continue
            visited.add(v)
            cost += w                            # add edge cost
            x, y = points[v]
            for i in range(n):                   # update edges
                if i not in visited:
                    xi, yi = points[i]
                    d = abs(x - xi) + abs(y - yi)
                    heapq.heappush(heap, (d, i))
        return cost                              # minimum total cost
# Time Complexity: O(N^2 log N)
# Space Complexity: O(N^2)

# Prim’s Algorithm (Minimum Spanning Tree)
# Works for connected, weighted, undirected graphs
import heapq
from typing import List
class Solution:
    def primMST(self, n: int, edges: List[List[int]]) -> int:
        adj = {i: [] for i in range(n)}            # adjacency list
        for u, v, w in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))
        visited = [False] * n                      # visited nodes
        heap = [(0, 0)]                            # (weight, node)
        mstCost = 0                                # total MST cost
        while heap:
            w, u = heapq.heappop(heap)
            if visited[u]:
                continue
            visited[u] = True                     # include node
            mstCost += w
            for v, wt in adj[u]:
                if not visited[v]:
                    heapq.heappush(heap, (wt, v))
        return mstCost
# Time Complexity: O(E log V)
# Space Complexity: O(V + E)

# Bellman-Ford Algorithm (Single Source Shortest Path)
# Works with negative weights and detects negative cycles
from typing import List
class Solution:
    def bellmanFord(self, V: int, edges: List[List[int]], src: int) -> List[int]:
        INF = 10**9
        dist = [INF] * V                           # distance array
        dist[src] = 0                              # source distance
        for _ in range(V - 1):                     # relax edges V-1 times
            for u, v, w in edges:
                if dist[u] != INF and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
        for u, v, w in edges:                      # check negative cycle
            if dist[u] != INF and dist[u] + w < dist[v]:
                return [-1]                        # negative cycle exists
        return dist
# Time Complexity: O(V * E)
# Space Complexity: O(V)

# 6. LeetCode 1282. Group the People Given the Group Size They Belong To
from typing import List
class Solution:
    def groupThePeople(self, groupSizes: List[int]) -> List[List[int]]:
        res = []                                  # final answer
        mp = {}                                   # size -> list of people
        for i in range(len(groupSizes)):                        # iterate each person
            size = groupSizes[i]                  # required group size
            if size not in mp:
                mp[size] = []                     # initialize list
            mp[size].append(i)                    # add person index
            if len(mp[size]) == size:             # group completed
                res.append(mp[size])              # add to result
                mp[size] = []                     # reset for next group
        return res
# Time Complexity: O(N)
# Space Complexity: O(N)

# ===================== PRIM'S ALGORITHM (MST) =====================
# Time: O(E log V) | Space: O(V+E) | Greedy: expand MST from a node
import heapq
def prim(V, adj):
    vis=[0]*V                      # mark nodes already in MST
    pq=[(0,0)]                     # (edge weight, node), start from 0
    mst=0                          # total MST weight
    while pq:
        w,u=heapq.heappop(pq)      # pick smallest edge
        if vis[u]: continue
        vis[u]=1; mst+=w           # include node and edge cost
        for v,wt in adj[u]:        # explore neighbors
            if not vis[v]:
                heapq.heappush(pq,(wt,v))
    return mst

# ===================== KRUSKAL'S ALGORITHM (MST) =====================
# Time: O(E log E) | Space: O(V+E) | Greedy: pick smallest edges
def kruskal(V, edges):
    parent=list(range(V)); rank=[0]*V   # DSU structures
    def find(x):                        # find root
        if parent[x]!=x: parent[x]=find(parent[x])
        return parent[x]
    def union(x,y):                     # union by rank
        rx,ry=find(x),find(y)
        if rx==ry: return False         # cycle detected
        if rank[rx]<rank[ry]: parent[rx]=ry
        else:
            parent[ry]=rx
            if rank[rx]==rank[ry]: rank[rx]+=1
        return True
    edges.sort()                        # sort edges by weight
    mst=0
    for w,u,v in edges:
        if union(u,v): mst+=w           # add edge if no cycle
    return mst

# ===================== DIJKSTRA'S ALGORITHM =====================
# Time: O(E log V) | Space: O(V) | No negative edges
def dijkstra(V, adj, src):
    dist=[10**8]*V; dist[src]=0          # distance array
    pq=[(0,src)]                         # min-heap by distance
    while pq:
        d,u=heapq.heappop(pq)
        if d>dist[u]: continue           # outdated entry
        for v,w in adj[u]:
            if dist[v]>d+w:              # relaxation step
                dist[v]=d+w
                heapq.heappush(pq,(dist[v],v))
    return dist

# ===================== BELLMAN FORD ALGORITHM =====================
# Time: O(VE) | Space: O(V) | Handles negative edges + detects cycles
def bellmanFord(V, edges, src):
    dist=[10**8]*V; dist[src]=0
    for _ in range(V-1):                 # relax all edges V-1 times
        for u,v,w in edges:
            if dist[u]!=10**8 and dist[v]>dist[u]+w:
                dist[v]=dist[u]+w
    for u,v,w in edges:                  # extra iteration for cycle
        if dist[u]!=10**8 and dist[v]>dist[u]+w:
            return "Negative Cycle"
    return dist

# ===================== FLOYD WARSHALL ALGORITHM =====================
# Time: O(V^3) | Space: O(V^2) | All-pairs shortest paths
def floydWarshall(dist):
    n=len(dist); INF=10**8
    for k in range(n):                   # intermediate node
        for i in range(n):               # source node
            for j in range(n):           # destination node
                if dist[i][k]!=INF and dist[k][j]!=INF:
                    dist[i][j]=min(dist[i][j],dist[i][k]+dist[k][j])

# ===================== KOSARAJU'S ALGORITHM (SCC) =====================
# Time: O(V+E) | Space: O(V+E) | Strongly Connected Components
def kosaraju(V, adj):
    vis=[0]*V; stack=[]
    def dfs(u):                          # first DFS (order)
        vis[u]=1
        for v in adj[u]:
            if not vis[v]: dfs(v)
        stack.append(u)
    for i in range(V):
        if not vis[i]: dfs(i)
    rev=[[] for _ in range(V)]           # reverse graph
    for u in range(V):
        for v in adj[u]: rev[v].append(u)
    vis=[0]*V; scc=0
    def dfs2(u):                         # second DFS
        vis[u]=1
        for v in rev[u]:
            if not vis[v]: dfs2(v)
    while stack:
        u=stack.pop()
        if not vis[u]:
            dfs2(u); scc+=1
    return scc

# ===================== BRIDGES IN GRAPH (TARJAN) =====================
# Time: O(V+E) | Space: O(V) | Critical edges
def bridges(V, adj):
    tin=[-1]*V; low=[-1]*V; timer=0; res=[]
    def dfs(u,p):
        nonlocal timer
        tin[u]=low[u]=timer; timer+=1     # discovery time
        for v in adj[u]:
            if v==p: continue
            if tin[v]!=-1:               # back edge
                low[u]=min(low[u],tin[v])
            else:
                dfs(v,u)
                low[u]=min(low[u],low[v])
                if low[v]>tin[u]:         # bridge condition
                    res.append((u,v))
    for i in range(V):
        if tin[i]==-1: dfs(i,-1)
    return res

# ===================== ARTICULATION POINT (TARJAN) =====================
# Time: O(V+E) | Space: O(V) | Critical vertices
def articulationPoints(V, adj):
    tin=[-1]*V; low=[-1]*V; timer=0; ap=[0]*V
    def dfs(u,p):
        nonlocal timer
        tin[u]=low[u]=timer; timer+=1
        child=0
        for v in adj[u]:
            if v==p: continue
            if tin[v]!=-1:               # back edge
                low[u]=min(low[u],tin[v])
            else:
                dfs(v,u)
                low[u]=min(low[u],low[v])
                if low[v]>=tin[u] and p!=-1:
                    ap[u]=1
                child+=1
        if p==-1 and child>1: ap[u]=1     # root condition
    for i in range(V):
        if tin[i]==-1: dfs(i,-1)
    return ap

# =====================================================================
# 1334. Find the City With the Smallest Number of Neighbors at a Threshold Distance | Floyd–Warshall
import math
class Solution:
    def findTheCity(self, n, edges, distanceThreshold):
        d = [[math.inf] * n for _ in range(n)]  # distance matrix
        for u, v, w in edges:
            d[u][v] = w  # edge u->v
            d[v][u] = w  # edge v->u
        for i in range(n):
            d[i][i] = 0  # self distance
        for k in range(n):  # intermediate
            for i in range(n):  # source
                for j in range(n):  # destination
                    if d[i][k] != math.inf and d[k][j] != math.inf:
                        d[i][j] = min(d[i][j], d[i][k] + d[k][j])
        cntCity = n  # minimum reachable count
        cityNo = -1  # answer
        for city in range(n):
            cnt = 0
            for adjCity in range(n):
                if d[city][adjCity] <= distanceThreshold:
                    cnt += 1
            if cnt <= cntCity:  # tie → larger index
                cntCity = cnt
                cityNo = city
        return cityNo
# -------- Sample Input --------
n = 4
edges = [[0,1,3],[1,2,1],[1,3,4],[2,3,1]]
distanceThreshold = 4
print(Solution().findTheCity(n, edges, distanceThreshold))  # Output: 3
# Time Complexity: O(n^3)
# Space Complexity: O(n^2)

# Get Connected Components in an Undirected Graph | DFS
# Time Complexity: O(V + E) → each vertex and edge visited once
# Space Complexity: O(V + E) → adjacency list + visited + recursion stack
from collections import defaultdict
class Solution:
    def getComponents(self, V, edges):
        res = []  # stores all components
        vis = [False] * V  # visited array
        g = defaultdict(list)  # adjacency list
        for u, v in edges:
            g[u].append(v)  # add edge u->v
            g[v].append(u)  # add edge v->u
        def dfs(u, comp):
            vis[u] = True  # mark node visited
            comp.append(u)  # add node to component
            for v in g[u]:
                if not vis[v]:
                    dfs(v, comp)
        for i in range(V):
            if not vis[i]:  
                comp = []
                dfs(i, comp)
                res.append(comp)
        return res
# Algorithm Steps:
# 1. Build adjacency list from edges
# 2. Maintain visited array
# 3. Run DFS from each unvisited node
# 4. Collect nodes as one connected component
# 5. Continue until all nodes are visited

# 785. Is Graph Bipartite | BFS Coloring
# Time Complexity: O(V + E) → each node and edge processed once
# Space Complexity: O(V) → color array + queue
from collections import deque
class Solution:
    def isBipartite(self, graph):
        n = len(graph)
        color = [-1] * n  # -1 = uncolored, 0/1 = two colors
        for start in range(n):  # handle disconnected graph
            if color[start] == -1:
                q = deque([start])
                color[start] = 0  # initial color
                while q:
                    u = q.popleft()
                    for v in graph[u]:
                        if color[v] == -1:
                            color[v] = 1 - color[u]  # assign opposite color
                            q.append(v)
                        elif color[v] == color[u]:
                            return False  # same color on both ends
        return True
# Algorithm Steps:
# 1. Initialize color array with -1 (uncolored)
# 2. For each uncolored node, start BFS
# 3. Color start node with 0
# 4. Assign opposite color to neighbors
# 5. If conflict occurs, return False
# 6. If all nodes processed, return True
or
# Is Graph Bipartite | DFS Coloring
# Time Complexity: O(V + E) → each vertex and edge visited once
# Space Complexity: O(V) → color array + recursion stack
class Solution:
    def isBipartite(self, graph):
        n = len(graph)
        color = [-1] * n  # -1 = uncolored, 0/1 = colors
        def dfs(u):
            for v in graph[u]:
                if color[v] == -1:
                    color[v] = 1 - color[u]  # opposite color
                    if not dfs(v):
                        return False
                elif color[v] == color[u]:
                    return False  # conflict
            return True
        for i in range(n):  # handle disconnected graph
            if color[i] == -1:
                color[i] = 0  # start color
                if not dfs(i):
                    return False
        return True
# Algorithm Steps:
# 1. Initialize all nodes as uncolored
# 2. For each unvisited node, assign color 0
# 3. Run DFS and color neighbors with opposite color
# 4. If same-color conflict occurs, return False
# 5. If DFS finishes for all nodes, return True

# 200. Number of Islands | DFS
# Time Complexity: O(m * n) → each cell visited once
# Space Complexity: O(m * n) → recursion stack in worst case
class Solution:
    def numIslands(self, grid):
        m, n = len(grid), len(grid[0])
        def dfs(i, j):
            if i < 0 or j < 0 or i >= m or j >= n or grid[i][j] != "1":
                return
            grid[i][j] = "0"  # mark visited
            dfs(i + 1, j)  # down
            dfs(i - 1, j)  # up
            dfs(i, j + 1)  # right
            dfs(i, j - 1)  # left
        cnt = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == "1":
                    dfs(i, j)
                    cnt += 1
        return cnt
# Algorithm Steps:
# 1. Traverse each cell in the grid
# 2. When land ('1') is found, start DFS
# 3. Mark all connected land as water
# 4. Increment island count for each DFS start
# 5. Return total island count

# 994. Rotting Oranges | BFS (Multi-source)
# Time Complexity: O(m * n) → each cell processed once
# Space Complexity: O(m * n) → queue in worst case
from collections import deque
class Solution:
    def orangesRotting(self, grid):
        m, n = len(grid), len(grid[0])
        q = deque()
        fresh = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 2:
                    q.append((i, j, 0))  # rotten orange with time
                elif grid[i][j] == 1:
                    fresh += 1
        time = 0
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        while q:
            i, j, t = q.popleft()
            time = max(time, t)
            for di, dj in dirs:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n and grid[ni][nj] == 1:
                    grid[ni][nj] = 2  # rot it
                    fresh -= 1
                    q.append((ni, nj, t + 1))
        return time if fresh == 0 else -1
# Algorithm Steps:
# 1. Push all initially rotten oranges into queue
# 2. Count fresh oranges
# 3. BFS level by level to rot neighbors
# 4. Track time using BFS depth
# 5. If fresh remains, return -1 else return time

# 542. 01 Matrix | BFS (Multi-source)
# Time Complexity: O(m * n) → each cell processed once in BFS
# Space Complexity: O(m * n) → queue + visited + distance matrix
from collections import deque
class Solution:
    def updateMatrix(self, mat):
        m, n = len(mat), len(mat[0])
        vis = [[0] * n for _ in range(m)]  # visited matrix
        dist = [[0] * n for _ in range(m)]  # distance matrix
        q = deque()
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0:
                    q.append((i, j, 0))  # multi-source BFS
                    vis[i][j] = 1
        dr = [-1, 0, 1, 0]
        dc = [0, 1, 0, -1]
        while q:
            r, c, steps = q.popleft()
            dist[r][c] = steps
            for k in range(4):
                nr = r + dr[k]
                nc = c + dc[k]
                if 0 <= nr < m and 0 <= nc < n and vis[nr][nc] == 0:
                    vis[nr][nc] = 1
                    q.append((nr, nc, steps + 1))
        return dist
# Algorithm Steps:
# 1. Add all cells with value 0 to queue
# 2. Mark them visited with distance 0
# 3. Run BFS in 4 directions
# 4. Update distance using BFS level
# 5. Return distance matrix

# 210. Course Schedule II | DFS + Topological Sort
# Time Complexity: O(V + E) → each course and edge visited once
# Space Complexity: O(V + E) → graph + recursion stack
from collections import defaultdict
class Solution:
    def findOrder(self, numCourses, prerequisites):
        g = defaultdict(list)  # adjacency list
        for v, u in prerequisites:
            g[u].append(v)  # u -> v
        vis = [False] * numCourses  # visited
        recPath = [False] * numCourses  # recursion path
        stack = []  # topo stack
        def isCycleDFS(u):
            vis[u] = True
            recPath[u] = True
            for v in g[u]:
                if not vis[v]:
                    if isCycleDFS(v):
                        return True
                elif recPath[v]:
                    return True
            recPath[u] = False
            return False
        for i in range(numCourses):
            if not vis[i]:
                if isCycleDFS(i):
                    return []  # cycle found
        vis = [False] * numCourses
        def topoDFS(u):
            vis[u] = True
            for v in g[u]:
                if not vis[v]:
                    topoDFS(v)
            stack.append(u)  # postorder
        for i in range(numCourses):
            if not vis[i]:
                topoDFS(i)
        return stack[::-1]  # reverse topo order
# Algorithm Steps:
# 1. Build graph from prerequisites
# 2. Detect cycle using DFS + recursion stack
# 3. If cycle exists, return empty list
# 4. Run DFS-based topological sort
# 5. Reverse stack to get answer

# Alien Dictionary – DFS vs BFS (Topological Sort)

from collections import defaultdict, deque

# ===================== DFS APPROACH =====================
class SolutionDFS:
    def findOrder(self, words):
        adj = defaultdict(list)
        vis = {}
        rec = {}
        res = []
        self.cycle = False
        for w in words:
            for c in w:
                vis[c] = False
                rec[c] = False
        for i in range(len(words) - 1):
            s1, s2 = words[i], words[i + 1]
            if len(s1) > len(s2) and s1.startswith(s2):
                return ""
            for j in range(min(len(s1), len(s2))):
                if s1[j] != s2[j]:
                    adj[s1[j]].append(s2[j])
                    break
        def dfs(u):
            vis[u] = True
            rec[u] = True
            for v in adj[u]:
                if not vis[v]:
                    dfs(v)
                elif rec[v]:
                    self.cycle = True
            rec[u] = False
            res.append(u)
        for c in vis:
            if not vis[c]:
                dfs(c)
        if self.cycle:
            return ""
        return ''.join(res[::-1])

# Time Complexity: O(N * L)
# Space Complexity: O(K)
# Algorithm Steps:
# 1. Initialize graph, visited, and recursion stack.
# 2. Build graph from adjacent words.
# 3. Check invalid prefix condition.
# 4. Run DFS for topological sort.
# 5. Detect cycle using recursion stack.
# 6. Reverse DFS order for result.


# ===================== BFS APPROACH =====================
class SolutionBFS:
    def findOrder(self, words):
        adj = defaultdict(list)
        indeg = {}
        for w in words:
            for c in w:
                indeg[c] = 0
        for i in range(len(words) - 1):
            s1, s2 = words[i], words[i + 1]
            if len(s1) > len(s2) and s1.startswith(s2):
                return ""
            for j in range(min(len(s1), len(s2))):
                if s1[j] != s2[j]:
                    adj[s1[j]].append(s2[j])
                    indeg[s2[j]] += 1
                    break
        q = deque()
        for c in indeg:
            if indeg[c] == 0:
                q.append(c)
        res = []
        while q:
            u = q.popleft()
            res.append(u)
            for v in adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        if len(res) != len(indeg):
            return ""
        return ''.join(res)

# Time Complexity: O(N * L)
# Space Complexity: O(K)
# Algorithm Steps:
# 1. Build graph and indegree map.
# 2. Check invalid prefix case.
# 3. Push all zero-indegree nodes into queue.
# 4. Process nodes using BFS (Kahn’s algorithm).
# 5. If processed nodes < total nodes → cycle exists.
# 6. Return valid order if no cycle.

# ===================== Cheapest Flight Within K Stops =====================
# Problem: Find the cheapest price from src to dst with at most k stops using BFS.
# Approach: BFS with level tracking (each level = number of stops).
# Note: We allow revisiting nodes since cheaper paths may appear.

from collections import defaultdict, deque
import math

class Solution:
    def findCheapestPrice(self, n: int, flights: list[list[int]], src: int, dst: int, k: int) -> int:
        adj = defaultdict(list)
        for u, v, w in flights:
            adj[u].append((v, w))  # build adjacency list

        q = deque([(src, 0)])  # queue holds (node, total_cost)
        dist = [math.inf] * n  # minimum cost to reach each node
        dist[src] = 0
        level = 0  # number of stops made so far

        while q and level <= k:
            size = len(q)
            while size:
                u, cost = q.popleft()
                for v, w in adj[u]:
                    if dist[v] > cost + w:  # relax edge
                        dist[v] = cost + w
                        q.append((v, dist[v]))  # push updated node to queue
                size -= 1
            level += 1  # finished current level (stop count increases)

        return -1 if dist[dst] == math.inf else dist[dst]

# ===================== Complexity =====================
# Time Complexity: O(K * E)   where E = number of flights, K = max stops
# Space Complexity: O(N + E)  for adjacency list, queue, and distance array

# ===================== Algorithm Steps =====================
# 1. Build adjacency list from flights: adj[u] = list of (v, cost)
# 2. Initialize BFS queue with source node and cost 0
# 3. Initialize distance array with infinity, set dist[src] = 0
# 4. Traverse BFS level by level; each level = 1 stop
# 5. For each node, relax edges to neighbors; update dist if cheaper
# 6. Add relaxed neighbors back to queue
# 7. Stop BFS after level > k
# 8. Return -1 if dst unreachable, else return minimum cost

# ===================== Remove Stones to Minimize Remaining =====================
# Problem: Remove as many stones as possible where a stone can be removed
# if another stone exists in the same row or column.
# Approach: Disjoint Set Union (Union-Find)

from typing import List

class Solution:
    def removeStones(self, stones: List[List[int]]) -> int:
        n = len(stones)
        parent = [i for i in range(n)]
        rank = [1] * n

        # Find with path compression
        def find(i):
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]

        # Union by rank
        def union(i, j):
            ri, rj = find(i), find(j)
            if ri == rj:
                return
            if rank[ri] < rank[rj]:
                parent[ri] = rj
            elif rank[ri] > rank[rj]:
                parent[rj] = ri
            else:
                parent[rj] = ri
                rank[ri] += 1

        # Union stones sharing same row or column
        for i in range(n):
            for j in range(i + 1, n):
                if stones[i][0] == stones[j][0] or stones[i][1] == stones[j][1]:
                    union(i, j)

        # Count number of connected components
        groups = 0
        for i in range(n):
            if find(i) == i:
                groups += 1

        return n - groups


# ===================== Complexity =====================
# Time Complexity: O(N^2 * α(N))
# Space Complexity: O(N)

# ===================== Algorithm Steps =====================
# 1. Treat each stone as a node in DSU.
# 2. Union two stones if they share the same row or column.
# 3. Count the number of connected components (groups).
# 4. In each group, all but one stone can be removed.
# 5. Answer = total stones - number of groups.
or
# ===================== Remove Stones to Minimize Remaining =====================
# Problem: Remove as many stones as possible where a stone can be removed
# if another stone exists in the same row or column.
# Approach: DFS on graph components

from typing import List
from collections import defaultdict

class Solution:
    def removeStones(self, stones: List[List[int]]) -> int:
        n = len(stones)
        graph = defaultdict(list)

        # Build graph: connect stones sharing row or column
        for i in range(n):
            for j in range(i + 1, n):
                if stones[i][0] == stones[j][0] or stones[i][1] == stones[j][1]:
                    graph[i].append(j)
                    graph[j].append(i)

        visited = [False] * n

        # DFS to explore a connected component
        def dfs(u):
            visited[u] = True
            for v in graph[u]:
                if not visited[v]:
                    dfs(v)

        components = 0
        for i in range(n):
            if not visited[i]:
                dfs(i)
                components += 1

        return n - components


# ===================== Complexity =====================
# Time Complexity: O(N^2)
# Space Complexity: O(N^2) (graph) + O(N) (visited)

# ===================== Algorithm Steps =====================
# 1. Treat each stone as a node.
# 2. Build an undirected graph connecting stones in same row/column.
# 3. Use DFS to find connected components.
# 4. Each component can keep one stone; rest can be removed.
# 5. Answer = total stones - number of components.

 # ===================== 1976. Number of Ways to Arrive at Destination =====================
# Approach: Dijkstra + Path Counting

from typing import List
from collections import defaultdict
import heapq

class Solution:
    def countPaths(self, n: int, roads: List[List[int]]) -> int:
        MOD = 10**9 + 7                          # modulo value
        adj = defaultdict(list)                 # adjacency list

        for u, v, t in roads:                   # build graph
            adj[u].append((v, t))
            adj[v].append((u, t))

        dist = [float('inf')] * n               # shortest distance array
        ways = [0] * n                          # number of shortest paths
        dist[0] = 0                             # source distance = 0
        ways[0] = 1                             # one way to reach source

        pq = [(0, 0)]                           # min-heap (dist, node)

        while pq:
            d, u = heapq.heappop(pq)            # pop smallest distance
            if d > dist[u]:
                continue                        # skip outdated entry
            for v, w in adj[u]:                 # explore neighbors
                nd = d + w                      # new distance
                if nd < dist[v]:                # found shorter path
                    dist[v] = nd                # update distance
                    ways[v] = ways[u]           # inherit path count
                    heapq.heappush(pq, (nd, v)) # push updated state
                elif nd == dist[v]:              # found another shortest path
                    ways[v] = (ways[v] + ways[u]) % MOD

        return ways[n - 1]                      # ways to reach destination


# ===================== Complexity =====================
# Time Complexity: O(E log V)
# Space Complexity: O(V + E)

# ===================== Algorithm Steps =====================
# 1. Build graph using adjacency list.
# 2. Initialize distance and path count arrays.
# 3. Apply Dijkstra’s algorithm.
# 4. Update path counts when shortest distance is matched.
# 5. Return number of shortest paths to node n-1.

# ===================== LeetCode 329. Longest Increasing Path in a Matrix =====================
# Approach: DFS + Memoization (DP)

from typing import List

class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        m, n = len(matrix), len(matrix[0])       # matrix dimensions
        dp = {}                                 # cache: (r, c) -> longest path
        
        def dfs(r, c, prevVal):
            if r < 0 or r == m or c < 0 or c == n:   # out of bounds
                return 0
            if matrix[r][c] <= prevVal:             # not increasing
                return 0
            if (r, c) in dp:                         # memoized result
                return dp[(r, c)]
            res = 1                                 # path length starting here
            res = max(res, 1 + dfs(r - 1, c, matrix[r][c]))  # up
            res = max(res, 1 + dfs(r, c + 1, matrix[r][c]))  # right
            res = max(res, 1 + dfs(r + 1, c, matrix[r][c]))  # down
            res = max(res, 1 + dfs(r, c - 1, matrix[r][c]))  # left
            dp[(r, c)] = res                         # store in cache
            return res
        
        for i in range(m):
            for j in range(n):
                dfs(i, j, -1)                        # start DFS from each cell
        
        return max(dp.values())                      # maximum path length


# ===================== Complexity =====================
# Time Complexity: O(M * N)
# Space Complexity: O(M * N)

# ===================== Algorithm Steps =====================
# 1. Use DFS from each cell to explore increasing paths.
# 2. Stop DFS if out of bounds or value not increasing.
# 3. Use memoization to avoid recomputation.
# 4. Try all four directions from each cell.
# 5. Return the maximum value stored in DP.

# ===================== LeetCode 1928. Minimum Cost to Reach Destination in Time =====================
# Approach: Dijkstra (Min-Heap) with State = (cost, time, node)

from typing import List
from collections import defaultdict
import heapq
import math

class Solution:
    def minCost(self, maxTime: int, edges: List[List[int]], passingFees: List[int]) -> int:
        adj = defaultdict(list)
        for u, v, t in edges:
            adj[u].append((v, t))          # add edge u -> v
            adj[v].append((u, t))          # add edge v -> u

        n = len(passingFees)
        minTime = [math.inf] * n           # minimum time to reach each node
        pq = []                            # min-heap
        
        heapq.heappush(pq, (passingFees[0], 0, 0))  # (cost, time, node)
        minTime[0] = 0

        while pq:
            cost, time, node = heapq.heappop(pq)    # get min cost state
            if node == n - 1:
                return cost                         # reached destination
            for ngbr, t in adj[node]:
                newTime = time + t                  # total time
                newCost = cost + passingFees[ngbr]  # total cost
                if newTime <= maxTime and newTime < minTime[ngbr]:
                    minTime[ngbr] = newTime         # update best time
                    heapq.heappush(pq, (newCost, newTime, ngbr))

        return -1                                   # destination unreachable


# ===================== Complexity =====================
# Time Complexity: O(E log V)
# Space Complexity: O(V + E)

# ===================== Algorithm Steps =====================
# 1. Build adjacency list from edges.
# 2. Use a min-heap storing (cost, time, node).
# 3. Track minimum time to reach each node.
# 4. Always expand the path with minimum cost.
# 5. Skip paths exceeding maxTime.
# 6. Return cost when destination node is reached.
# 7. If unreachable within time, return -1.

# LeetCode 1489. Find Critical and Pseudo-Critical Edges in Minimum Spanning Tree
class Solution:
    def findCriticalAndPseudoCriticalEdges(self, n: int, edges: List[List[int]]) -> List[List[int]]:
        # Disjoint Set Union (Union-Find)
        class DSU:
            def __init__(self, n):
                self.parent = list(range(n))  # parent[i] = parent of node i
                self.rank = [0]*n  # rank for union by rank
            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])  # path compression
                return self.parent[x]
            def union(self, x, y):
                rx, ry = self.find(x), self.find(y)
                if rx == ry:
                    return False  # already connected
                if self.rank[rx] < self.rank[ry]:
                    self.parent[rx] = ry
                elif self.rank[rx] > self.rank[ry]:
                    self.parent[ry] = rx
                else:
                    self.parent[ry] = rx
                    self.rank[rx] += 1
                return True
        # Kruskal MST with option to skip or force an edge
        def kruskal(skipEdge, addEdge):
            dsu = DSU(n)  # fresh DSU for each MST run
            total = 0  # total weight of MST
            if addEdge != -1:
                u,v,w,_ = edges[addEdge]
                if dsu.union(u,v):
                    total += w  # force include this edge
            for i in range(len(edges)):
                if i == skipEdge:
                    continue  # skip this edge
                u,v,w,_ = edges[i]
                if dsu.union(u,v):
                    total += w  # add edge if it connects components
            root = dsu.find(0)
            for i in range(1,n):
                if dsu.find(i) != root:
                    return float('inf')  # graph not fully connected
            return total
        # Append original index to each edge
        for i in range(len(edges)):
            edges[i].append(i)  # [u,v,w,idx]
        edges.sort(key=lambda x:x[2])  # sort by weight
        mstWeight = kruskal(-1,-1)  # base MST weight
        critical = []  # list of critical edge indices
        pseudo = []  # list of pseudo-critical edge indices
        for i in range(len(edges)):
            if kruskal(i,-1) > mstWeight:
                critical.append(edges[i][3])  # removing edge increases MST weight
            elif kruskal(-1,i) == mstWeight:
                pseudo.append(edges[i][3])  # forcing edge still gives MST
        return [critical,pseudo]
# Algo Steps:
# 1. Build MST using Kruskal to get base MST weight
# 2. For each edge:
#    a) Skip it → if MST weight increases, edge is critical
#    b) Force it → if MST weight stays same, edge is pseudo-critical
# Time Complexity: O(E * α(V) * E) ≈ O(E^2)
# Space Complexity: O(V)

