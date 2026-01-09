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
    # check whether a loop or not   
    def cloop(self):
        f=s=self.head
        while f and f.next:
            f=f.next.next
            s=s.next
            if f==s:
                print("loop")
                return
        print("no loop")
    # loop starting point 
    def loop_startpoint(self):
        f=s=self.head
        while f and f.next:
            f=f.next.next
            s=s.next
            if f==s:
                s=self.head
                while s!=f:
                    s=s.next
                    f=f.next
                print(s.data)   
    def loop_count(self):
        f=s=self.head
        while f and f.next:
            f=f.next.next
            s=s.next
            if f==s:
                s=s.next
                c=1
                while f!=s:
                    s=s.next
                    c+=1
                print(c)
                break
    # remove cycle
    def del_cycle(self):
        f=s=self.head
        while f and f.next:
            f=f.next.next
            s=s.next
            if f==s:
                break
        else:
            return "no loop"
        s=self.head
        while f!=s:
            p=f
            s=s.next
            f=f.next
        # p=p.next
        p.next=None
    # reverse ll
    def reversel(self):
        p=None
        c=self.head
        while c:
            n=c.next
            c.next=p
            p=c
            c=n
        self.head=p
    # palindrome
    def palindrome(self):
        f=s=self.head
        while f and f.next:
            f=f.next.next
            pr=s
            s=s.next
        p=None
        c=s
        while c:
            n=c.next
            c.next=p
            p=c
            c=n
        pr.next=p
        t2=p
        t1=self.head
        while  t2:
            if t1.data!=t2.data:
                print("False")
                return
            t1=t1.next
            t2=t2.next
        print("True")
    # intersection point of two linked lists
    def intersectionp(self,l1,l2):
        t1=l1.head
        t2=l2.head
        while t1!=t2:
            if t1!=None:
                t1=t1.next
            else:
                t1=l2.head
            if t2!=None:
                t2=t2.next
            else:
                t2=l1.head
        if t1:
            print(t1.data) 
        else:
            print("No intersection")
    # merge two sorted lists
    def merge(self,t1,t2):
        dummy=Node(-1)
        current=dummy
        while t1 and t2:
            if t1.data<t2.data:
                current.next=t1
                t1=t1.next
            else:
                current.next=t2
                t2=t2.next
            current=current.next
        current.next = t1 if t1 else t2
        # return dummy.next
        t=dummy.next
        while t:
            print(t.data,end="->")
            t=t.next
        print("None")
# l=Linked()
# l.head=Node(100)
# l.head.next=Node(200)
# l.head.next.next=Node(200)
# l.head.next.next.next=Node(100)
# l.head.next.next.next.next=Node(100)
# l.head.next.next.next.next.next=Node(600)
# l.head.next.next.next.next.next.next=l.head.next.next
# l.display()
# l.cloop()         
# l.loop_startpoint()
# l.loop_count()
# l.del_cycle()
# l.reversel()
# l.palindrome()
# l.display()
l1=Linked()
l1.add_back(10)
l1.add_back(20)
l1.add_back(30)
l1.add_back(40)
l1.add_back(50)
l1.display()
l2=Linked()
l2.add_back(100)
l2.add_back(200)
l2.add_back(300)
l2.add_back(400)
l2.add_back(500)
# l1.add_back(30)
# l1.add_back(40)
# l2.add_back(50)
l2.display()
l3=Linked()
# l3.intersectionp(l1,l2) # not working correctly
l3.merge(l1.head,l2.head)
l3.display()

#Add two numbers
