# from collections import deque

# class Node:
#     def __init__(self, data):
#         self.data = data
#         self.left = None
#         self.right = None

# class AVL:
#     def __init__(self):
#         self.node = None

#     def height(self, node):
#         if node is None:
#             return -1
#         return max(self.height(node.left), self.height(node.right)) + 1

#     def balfac(self, node):
#         return self.height(node.left) - self.height(node.right)

#     def left_rot(self, z):
#         y = z.right
#         T2 = y.left
#         y.left = z
#         z.right = T2
#         return y

#     def right_rot(self, z):
#         y = z.left
#         T2 = y.right
#         y.right = z
#         z.left = T2
#         return y

#     def insert(self, node, key):
#         if not node:
#             return Node(key)
#         elif key < node.data:
#             node.left = self.insert(node.left, key)
#         elif key > node.data:
#             node.right = self.insert(node.right, key)
#         else:
#             return node

#         balance = self.balfac(node)

#         if balance > 1 and key < node.left.data:
#             return self.right_rot(node)
#         if balance < -1 and key > node.right.data:
#             return self.left_rot(node)
#         if balance > 1 and key > node.left.data:
#             node.left = self.left_rot(node.left)
#             return self.right_rot(node)
#         if balance < -1 and key < node.right.data:
#             node.right = self.right_rot(node.right)
#             return self.left_rot(node)

#         return node

#     def insert_key(self, key):
#         self.node = self.insert(self.node, key)

#     def get_max_value_node(self, node):
#         temp = node
#         while temp.right:
#             temp = temp.right
#         return temp

#     def delete(self, node, key):
#         if not node:
#             return node
#         elif key < node.data:
#             node.left = self.delete(node.left, key)
#         elif key > node.data:
#             node.right = self.delete(node.right, key)
#         else:
#             if not node.left or not node.right:
#                 node = node.left or node.right
#             else:
#                 temp = self.get_max_value_node(node.left)
#                 node.data = temp.data
#                 node.left = self.delete(node.left, temp.data)

#         if not node:
#             return node

#         balance = self.balfac(node)

#         if balance > 1 and self.balfac(node.left) >= 0:
#             return self.right_rot(node)
#         if balance > 1 and self.balfac(node.left) < 0:
#             node.left = self.left_rot(node.left)
#             return self.right_rot(node)
#         if balance < -1 and self.balfac(node.right) <= 0:
#             return self.left_rot(node)
#         if balance < -1 and self.balfac(node.right) > 0:
#             node.right = self.right_rot(node.right)
#             return self.left_rot(node)

#         return node

#     def display(self):
#         if not self.node:
#             return
#         queue = deque([self.node])
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

# # 🚀 DEMO
# tree = AVL()
# for val in [30, 20, 40, 10, 25, 50, 5]:
#     tree.insert_key(val)

# print("Before Deletion:")
# tree.display()

# tree.node = tree.delete(tree.node, 50)

# print("\nAfter Deletion:")
# tree.display()


# find no of txt files in your file system
# import os
# def count_txt_files(root_path=None):
#     if root_path is None:
#         root_path = os.path.join(os.path.expanduser("~"), "Desktop")
#     count = 0
#     for dirpath, dirnames, filenames in os.walk(root_path):
#         #print(dirpath, dirnames, filenames)
#         count += sum(1 for file in filenames if file.endswith(".txt"))
#     return count
# print("Number of .txt files:", count_txt_files())

# #java
# import java.io.File;
# public class TxtFileCounter {
#     public static int countTxtFiles(File dir) {
#         int count = 0;
#         File[] files = dir.listFiles();
#         if (files == null) return 0;
#         for (File file : files) {
#             if (file.isDirectory()) {
#                 count += countTxtFiles(file);
#             } else if (file.getName().endsWith(".txt")) {
#                 count++;
#             }
#         }
#         return count;
#     }
#     public static void main(String[] args) {
#         File root = new File("/"); // Change to "C:\\" for Windows
#         int totalTxtFiles = countTxtFiles(root);
#         System.out.println("Total .txt files: " + totalTxtFiles);
#     }
# }

# from abc import ABC, abstractmethod
# from typing import List, Tuple, Optional

# class UndirectedGraphs(ABC):
#     @abstractmethod
#     def sendAutoAcceptedFriendRequest(self, x: int, y: int): pass  # O(1)
#     @abstractmethod
#     def removalFriendship(self, x: int, y: int): pass              # O(1)
#     @abstractmethod
#     def isFriend(self, x: int, y: int) -> bool: pass               # O(1)
#     @abstractmethod
#     def getFriends(self, x: int) -> List[int]: pass                # O(k), k = no. of friends

# class DirectedGraphs(ABC):
#     @abstractmethod
#     def follow(self, fan: int, celebrity: int): pass               # O(1)
#     @abstractmethod
#     def unfollow(self, fan: int, celebrity: int): pass             # O(1)
#     @abstractmethod
#     def isFollowing(self, fan: int, celebrity: int) -> bool: pass  # O(1)
#     @abstractmethod
#     def getFollowers(self, celebrity: int) -> List[int]: pass      # O(k), k = no. of followers

# class WeightedGraphs(ABC):
#     @abstractmethod
#     def establishRoute(self, source: int, destination: int, tollPrice: int): pass  # O(1)
#     @abstractmethod
#     def reportRoadDamaged(self, source: int, destination: int): pass              # O(1)
#     @abstractmethod
#     def getTollPrice(self, source: int, destination: int) -> Optional[int]: pass  # O(1)
#     @abstractmethod
#     def getAllRoutes(self, source: int) -> List[Tuple[int, int]]: pass            # O(k), k = routes from source

# class SocialGraph(UndirectedGraphs):
#     def __init__(self):
#         self.friends = {}  # Dict[int, Set[int]]
#     def sendAutoAcceptedFriendRequest(self, x, y):  # O(1)
#         self.friends.setdefault(x, set()).add(y)
#         self.friends.setdefault(y, set()).add(x)
#     def removalFriendship(self, x, y):  # O(1)
#         self.friends.get(x, set()).discard(y)
#         self.friends.get(y, set()).discard(x)
#     def isFriend(self, x, y) -> bool:  # O(1)
#         return y in self.friends.get(x, set())
#     def getFriends(self, x) -> List[int]:  # O(k)
#         return list(self.friends.get(x, set()))

# class FollowerGraph(DirectedGraphs):
#     def __init__(self):
#         self.following = {}  # Dict[int, Set[int]]
#         self.followers = {}  # Dict[int, Set[int]]
#     def follow(self, fan, celebrity):  # O(1)
#         self.following.setdefault(fan, set()).add(celebrity)
#         self.followers.setdefault(celebrity, set()).add(fan)
#     def unfollow(self, fan, celebrity):  # O(1)
#         self.following.get(fan, set()).discard(celebrity)
#         self.followers.get(celebrity, set()).discard(fan)
#     def isFollowing(self, fan, celebrity) -> bool:  # O(1)
#         return celebrity in self.following.get(fan, set())
#     def getFollowers(self, celebrity) -> List[int]:  # O(k)
#         return list(self.followers.get(celebrity, set()))

# class RoadNetwork(WeightedGraphs):
#     def __init__(self):
#         self.routes = {}  # Dict[int, Dict[int, int]]
#     def establishRoute(self, source, destination, tollPrice):  # O(1)
#         self.routes.setdefault(source, {})[destination] = tollPrice
#         self.routes.setdefault(destination, {})[source] = tollPrice
#     def reportRoadDamaged(self, source, destination):  # O( 1)
#         self.routes.get(source, {}).pop(destination, None)
#         self.routes.get(destination, {}).pop(source, None)
#     def getTollPrice(self, source, destination) -> Optional[int]:  # O(1)
#         return self.routes.get(source, {}).get(destination)
#     def getAllRoutes(self, source) -> List[Tuple[int, int]]:  # O(k)
#         return list(self.routes.get(source, {}).items())


# if __name__ == "__main__":
#     sg = SocialGraph()
#     sg.sendAutoAcceptedFriendRequest(1, 2)
#     sg.sendAutoAcceptedFriendRequest(1, 3)
#     print("Friends of 1:", sg.getFriends(1))
#     print("Is 1 friend with 2?", sg.isFriend(1, 2))
#     sg.removalFriendship(1, 2)
#     print("Is 1 friend with 2 after removal?", sg.isFriend(1, 2))

#     fg = FollowerGraph()
#     fg.follow(101, 200)
#     fg.follow(102, 200)
#     print("Followers of 200:", fg.getFollowers(200))
#     print("Is 101 following 200?", fg.isFollowing(101, 200))
#     fg.unfollow(101, 200)
#     print("Is 101 following 200 after unfollow?", fg.isFollowing(101, 200))

#     rn = RoadNetwork()
#     rn.establishRoute(10, 20, 50)
#     rn.establishRoute(10, 30, 70)
#     print("Routes from 10:", rn.getAllRoutes(10))
#     print("Toll from 10 to 20:", rn.getTollPrice(10, 20))
#     rn.reportRoadDamaged(10, 20)
#     print("Toll from 10 to 20 after damage:", rn.getTollPrice(10, 20))

# from collections import deque

# def bfs(graph, start):
#     visited = set()
#     queue = deque([start])
    
#     while queue:
#         node = queue.popleft()
#         if node not in visited:
#             print(node, end=' ')
#             visited.add(node)
#             queue.extend(graph[node])

# def dfs(graph, node, visited=None):
#     if visited is None:
#         visited = set()
#     print(node, end=' ')
#     visited.add(node)
#     for neighbor in graph[node]:
#         if neighbor not in visited:
#             dfs(graph, neighbor, visited)

# graph = {
#     'A': ['B', 'C'],
#     'B': ['D', 'E'],
#     'C': ['F'],
#     'D': [],
#     'E': ['F'],
#     'F': []
# }
# bfs(graph, 'A')
# dfs(graph, 'A')

# from collections import deque

# def bfs_matrix(graph, start):
#     visited = [False] * len(graph)
#     queue = deque([start])
#     visited[start] = True
#     print("BFS Traversal:", end=" ")
#     while queue:
#         node = queue.popleft()
#         print(node, end=' ')
#         for i, connected in enumerate(graph[node]):
#             if connected == 1 and not visited[i]:
#                 visited[i] = True
#                 queue.append(i)
#     print()

# def dfs_matrix(graph, node, visited=None):
#     if visited is None:
#         visited = [False] * len(graph)
#         print("DFS Traversal:", end=" ")
#     visited[node] = True
#     print(node, end=' ')
#     for i, connected in enumerate(graph[node]):
#         if connected == 1 and not visited[i]:
#             dfs_matrix(graph, i, visited)

# graph = [
#     [0, 1, 1, 0, 0, 0],  # Node 0 connected to 1, 2
#     [0, 0, 0, 1, 1, 0],  # Node 1 connected to 3, 4
#     [0, 0, 0, 0, 0, 1],  # Node 2 connected to 5
#     [0, 0, 0, 0, 0, 0],  # Node 3
#     [0, 0, 0, 0, 0, 1],  # Node 4 connected to 5
#     [0, 0, 0, 0, 0, 0]   # Node 5
# ]
# bfs_matrix(graph, 0)
# dfs_matrix(graph, 0)
