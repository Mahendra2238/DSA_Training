#  dont call as pointer variable call it as object variable in linkedlist as object is  being stored in address part of node
# Linked lists
# class Node:
#     def __init__(self,data):
#         self.data=data
#         self.next=None
# class Linked:
#     def __init__(self):
#         self.head=None
#     def add_back(self,x):
#         newnode=Node(x)
#         if self.head is None:
#             self.head = newnode
#         else:
#             t=self.head
#             while(t.next!=None):
#                 t=t.next
#             t.next=Node(x)
#     def display(self):
#         t=self.head
#         while(t!=None):
#             print(t.data,end='->')
#             t=t.next
#         print("None")
#     def sumall(self):
#         s=0
#         t=self.head
#         while t!=None:
#             s+=t.data
#             t=t.next
#         print("All sum: ",s)
#     def sum_even(self):
#         es=0
#         t=self.head
#         while t!=None:
#             if t.data%2==0:
#                 es+=t.data
#             t=t.next
#         print("Even sum: ",es)
#     def sum_odd(self):
#         os=0
#         t=self.head
#         while t!=None:
#             if t.data%2!=0:
#                 os+=t.data
#             t=t.next
#         print("Odd sum: ",os)
#     def sum_even_pos(self):
#         eps=0
#         t=self.head
#         pos=1
#         while t!=None:
#             if pos%2==0:
#                 eps+=t.data
#             t=t.next
#             pos+=1
#         print("Even position sum: ",eps)
#     def sec_largest(self):
#         if not self.head or not self.head.next:
#             print("Insufficient nodes")
#             return
#         fm=sm=0
#         t = self.head
#         while t:
#             if t.data > fm:
#                 sm = fm
#                 fm = t.data
#             elif fm > t.data > sm:
#                 sm = t.data
#             t = t.next
#         print("Second largest:", sm)
#     # consecutive pair sum whose sum<=k
#     def countcp_target(self, k):
#         t = self.head
#         c = 0
#         while t and t.next:
#             if t.data + t.next.data <= k:
#                 c += 1
#             t = t.next
#         print("Consecutive pair count:", c)
#     def count_allpair_target(self,k):
#         t=self.head
#         c=0
#         while t!=None:
#             t1=t.next
#             while t1!=None:
#                 if t.data+t1.data<=k:
#                     c+=1
#                 t1=t1.next
#             t=t.next
#         print("All concecutive pairs sum <=target: ",c)
# l=Linked()
# l.add_back(10)
# l.add_back(20)
# l.add_back(31)
# l.add_back(9)
# l.add_back(30)
# l.display()
# l.sumall()
# l.sum_even()
# l.sum_odd()
# l.sum_even_pos()
# l.sec_largest()
# l.countcp_target(30)
# l.count_allpair_target(30)

# print second half of sorted linked list
class Node:
    def __init__(self,data):
        self.data=data
        self.next=None
class Linked:
    def __init__(self):
        self.head=None
    def add_back(self,x):
        newnode=Node(x)
        if self.head is None:
            self.head = newnode
        else:
            t=self.head
            while(t.next!=None):
                t=t.next
            t.next=Node(x)
    def display(self):
        t=self.head
        while(t!=None):
            print(t.data,end='->')
            t=t.next
        print("None")
    def sec_half(self):
        f=s=self.head
        while f!=None and f.next!=None:
            f=f.next.next
            s=s.next
        # print(s.data)
        while s!=None:
            print(s.data,end='->')
            s=s.next
        print("None")
        # # o(n+n/2)
        # p=0
        # t=self.head
        # while t!=None:
        #     p+=1
        #     t=t.next
        # t=self.head
        # p=p//2
        # m=0
        # while m!=p:
        #     t=t.next
        #     m+=1
        # while t!=None:
        #     print(t.data,end='->')
        #     t=t.next
        # print("None")
    # floyeds 
    def mid(self):
        f=s=self.head
        while f!=None and f.next!=None:
            f=f.next.next
            s=s.next
        print(s.data)
    def klast(self,k):
        f=s=self.head
        for i in range(k):
            f=f.next
        while f!=None:
            f=f.next
            s=s.next
        print(s.data)
        # p=0
        # t=self.head
        # while t!=None:
        #     p+=1
        #     t=t.next
        # t=self.head
        # p=p-k
        # m=0
        # while m!=p:
        #     t=t.next
        #     m+=1
        # print(t.data)
    def klast_del(self,k):
        f=s=self.head
        for i in range(k):
            f=f.next
        while f!=None:
            prev=s
            s=s.next
            f=f.next
        print("Element deleted: ",s.data)
        prev.next=s.next    
    def swap(self):
        t=self.head
        while t!=None and t.next!=None:
            t.data,t.next.data=t.next.data,t.data
            t=t.next.next  
    def bubblesort(self):
        end=None
        while end!=self.head:
            t=self.head
            f=0
            while t.next!=end:
                if t.data>t.next.data:
                    t.data,t.next.data=t.next.data,t.data
                    f=1
                t=t.next
            end=t
            if f==0:
                break
    # kth largest element in the unsorted linkedlist
    def klargest(self,k):
        k1=k
        t=self.head
        while t!=None and k!=0:
            t=self.head
            f=0
            while t.next!=None:
                if t.data>t.next.data:
                    t.data,t.next.data=t.next.data,t.data
                    f=1
                t=t.next
            if f==0:
                break
            k=k-1
        self.klast(k1)
l=Linked()
l.add_back(10)
l.add_back(80)
l.add_back(30)
l.add_back(90)
l.add_back(40)
l.add_back(50)
l.add_back(60)
l.display()
# l.sec_half()
# l.mid()
# l.klast(2)
# l.klast_del(2)
# l.display()
# l.display()
# l.swap()
l.bubblesort()
l.display()
l.klargest(6)

# check even or odd use fast pointer jump double if it is at lastnode odd or at none it is even
# loop

