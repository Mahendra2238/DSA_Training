# # 206. Reverse Linked List
# class Solution:
#     def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
#         p=None
#         c=head
#         while c:
#             n=c.next
#             c.next=p
#             p=c
#             c=n
#         return p

# # 876. Middle of the Linked List
# class Solution:
#     def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
#         slow=head
#         fast=head
#         while fast!=None and fast.next!=None:
#             slow=slow.next
#             fast=fast.next.next
#         return slow

# # 21. Merge Two Sorted Lists
# def mergeTwoLists(h1,h2):
#     if h1 is None or h2 is None:
#         return h2 if h1==None else h1
#     if h1.val<=h2.val:
#         h1.next=mergeTwoLists(h1.next,h2)
#         return h1
#     else:
#         h2.next=mergeTwoLists(h1,h2.next)
#         return h2
# list1 = [1,2,4]
# list2 = [1,3,4]
# from linkedlist import create_linked_list,print_linked_list
# h1 = create_linked_list(list1)
# h2 = create_linked_list(list2)
# res = mergeTwoLists(h1,h2)
# print_linked_list(res)

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

# 234. Palindrome Linked List
# class Solution:
#     def isPalindrome(self, head: Optional[ListNode]) -> bool:
#         def reverse(head):
#             prev = None
#             cur = head
#             while cur:
#                 nxt = cur.next
#                 cur.next = prev
#                 prev = cur
#                 cur = nxt
#             return prev
#         slow=head
#         fast=head
#         # find middle
#         while fast.next is not None and fast.next.next is not None:
#             slow=slow.next
#             fast=fast.next.next
#         # reverse second half
#         newhead=reverse(slow.next)
#         first=head
#         second=newhead
#         # compare halves
#         while second!=None:
#             if first.val!=second.val:
#                 return False
#             first=first.next
#             second=second.next
#         return True

# 141. Linked List Cycle
# class Solution:
#     def hasCycle(self, head: Optional[ListNode]) -> bool:
#         f=s=head
#         while f and f.next:
#             f=f.next.next
#             s=s.next
#             if f==s:
#                 return True
#         return False

# 142. Linked List Cycle II
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# class Solution:
#     # detect starting node point of cycle in linked list
#     def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
#         f=s=head
#         while f and f.next:
#             f=f.next.next
#             s=s.next
#             if f==s:
#                 s=head
#                 while s!=f:
#                     s=s.next
#                     f=f.next
#                 return s
#         return None

#         # remove cycle
#         # while f and f.next:
#         #     f=f.next.next
#         #     s=s.next
#         #     if f==s:
#         #         s=head
#         #         p=None
#         #         while s!=f:
#         #             s=s.next
#         #             p=f
#         #             f=f.next
#         #         p.next=None //remove cycle (tail)
#         #         return s
#         # return None

"""
# Definition for a Node.
class Node:
    def __init__(self, val, prev, next, child):
        self.val = val
        self.prev = prev
        self.next = next
        self.child = child
"""

# 430. Flatten a Multilevel Doubly Linked List
# class Solution:
#     def flatten(self, head: 'Optional[Node]') -> 'Optional[Node]':
#         if head==None:
#             return head
#         curr=head
#         while curr!=None:
#             # Flatten the child nodes
#             if curr.child!=None:# if child exists
#                 nxt=curr.next
#                 curr.next=self.flatten(curr.child)  # child head
#                 curr.next.prev=curr
#                 curr.child=None
#                 # Find tail (of child nodes attched)
#                 while curr.next:
#                     curr=curr.next
#                 # Attach tail with next ptr
#                 if nxt:
#                     curr.next=nxt
#                     nxt.prev=curr
#             curr=curr.next
        # return head

"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""

# 138. Copy List with Random Pointer
# class Solution:
#     def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
#         if not head:
#             return None
#         m={}
#         newHead=Node(head.val)
#         oldTemp=head.next
#         newTemp=newHead
#         m[head]=newHead
#         while oldTemp is not None:
#             copyNode=Node(oldTemp.val)
#             m[oldTemp]=copyNode
#             newTemp.next=copyNode
#             oldTemp=oldTemp.next
#             newTemp=newTemp.next
#         oldTemp=head
#         newTemp=newHead
#         while oldTemp is not None:
#             if oldTemp.random is None:
#                 newTemp.random=None
#             else:
#                 newTemp.random=m[oldTemp.random]
#             oldTemp=oldTemp.next
#             newTemp=newTemp.next
#         return newHead

# 2. Add Two Numbers
# def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
#         t3=ListNode(0)
#         curr=t3
#         c=0
#         while l1 or l2 or c:
#             x=l1.val if l1 else 0
#             y=l2.val if l2 else 0
#             s=x+y+c
#             c=s//10
#             curr.next=ListNode(s%10)
#             curr=curr.next
#             if l1:
#                 l1=l1.next
#             if l2:
#                 l2=l2.next
#         return t3.next

# 92. Reverse Linked List II
# def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
#     dummy=ListNode(-1)
#     dummy.next=head
#     leftPre=dummy # marker before left
#     curr=head
#     for _ in range(left-1):
#         leftPre=leftPre.next
#         curr=curr.next
#     subListHead=curr # marker of starting point of rev list part
#     # reverse logic
#     prev=None
#     for _ in range(right-left+1): # r-l+1 times loop   
#         next=curr.next
#         curr.next=prev
#         prev=curr
#         curr=next
#     leftPre.next=prev
#     subListHead.next=curr
#     return dummy.next

# 146. LRU Cache
# class LRUCache:
#     class Node:
#         def __init__(self,k,v):
#             self.key=k
#             self.val=v
#             self.prev=None
#             self.next=None

#     def __init__(self, capacity: int):
#         self.limit = capacity
#         self.m = {} 
#         self.head = self.Node(-1, -1)
#         self.tail = self.Node(-1, -1)
#         self.head.next = self.tail
#         self.tail.prev = self.head
    
#     def addNode(self,newNode):
#         oldNext=self.head.next
        
#         self.head.next=newNode
#         oldNext.prev=newNode

#         newNode.next=oldNext
#         newNode.prev=self.head

#     def delNode(self,oldNode):
#         oldPrev=oldNode.prev
#         oldNext=oldNode.next

#         oldPrev.next=oldNext
#         oldNext.prev=oldPrev

#     def get(self, key: int) -> int:
#         if key not in self.m:
#             return -1
#         ansNode=self.m[key]
#         ans=ansNode.val

#         del self.m[key]
#         self.delNode(ansNode)

#         self.addNode(ansNode)  # update mru
#         self.m[key]=ansNode

#         return ans

#     def put(self, key: int, value: int) -> None:
#         if key in self.m:
#             oldNode=self.m[key]
#             self.delNode(oldNode)
#             del self.m[key]

#         if len(self.m)==self.limit:
#             del self.m[self.tail.prev.key]
#             self.delNode(self.tail.prev)

#         newNode=self.Node(key,value)
#         self.addNode(newNode)
#         self.m[key]=newNode

# # Your LRUCache object will be instantiated and called as such:
# # obj = LRUCache(capacity)
# # param_1 = obj.get(key)
# # obj.put(key,value)

# # 61. Rotate List
# def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
#         def findNthNode(temp,k):
#             cnt=1
#             while temp is not None:
#                 if cnt==k:
#                     return temp
#                 cnt+=1
#                 temp=temp.next
#             return temp
#         if head==None or k==0:
#             return head
#         tail=head
#         l=1 # length
#         while tail.next is not None:
#             tail=tail.next
#             l+=1
#         if k%l == 0: return head
#         k=k%l  #14 -> 12%5=2  0r 15 -> (5*5*5) 15%5=0
#         # attach tail to head
#         tail.next=head
#         # update head
#         newLastNode=findNthNode(head,l-k)
#         head=newLastNode.next
#         newLastNode.next=None
#         return head

# 25. Reverse Nodes in k-Group
# class Solution:
#     def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
#         # check if k nodes exists
#         c=0
#         temp=head
#         while c<k:
#             if temp==None:
#                 return head
#             temp=temp.next
#             c+=1
#         # recursively call rest of ll
#         prevNode=self.reverseKGroup(temp,k)
#         # reverse current group
#         temp=head
#         cnt=0
#         while cnt<k:
#             next=temp.next
#             temp.next=prevNode
#             prevNode=temp
#             temp=next
#             cnt+=1
#         return prevNode

# # 21. Merge Two Sorted Lists
# def mergeTwoLists(h1,h2):
#     if h1 is None or h2 is None:
#         return h2 if h1==None else h1
#     if h1.val<=h2.val:
#         h1.next=mergeTwoLists(h1.next,h2)
#         return h1
#     else:
#         h2.next=mergeTwoLists(h1,h2.next)
#         return h2
# list1 = [1,2,4]
# list2 = [1,3,4]
# from linkedlist import create_linked_list,print_linked_list
# h1 = create_linked_list(list1)
# h2 = create_linked_list(list2)
# res = mergeTwoLists(h1,h2)
# print_linked_list(res)
