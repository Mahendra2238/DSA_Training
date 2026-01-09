# 225. Implement Stack using Queues
from collections import deque
class MyStack:
    def __init__(self):
        self.q1=deque()
        self.q2=deque()
    def push(self, x: int) -> None:
        while self.q1:
            self.q2.append(self.q1.popleft())
        self.q1.append(x)
        while self.q2:
            self.q1.append(self.q2.popleft())
    def pop(self) -> int:
        return self.q1.popleft()
    def top(self) -> int:
        return self.q1[0]
    def empty(self) -> bool:
        return len(self.q1)==0
# Your MyStack object will be instantiated and called as such:
# obj = MyStack()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.top()
# param_4 = obj.empty()

# 232. Implement Queue using Stacks
class MyQueue:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []
    def push(self, x: int) -> None:
        # Move all from stack1 -> stack2
        while self.stack1:
            self.stack2.append(self.stack1.pop())
        # Insert new element
        self.stack1.append(x)
        # Move back stack2 -> stack1
        while self.stack2:
            self.stack1.append(self.stack2.pop())
    def pop(self) -> int:
        return self.stack1.pop()
    def peek(self) -> int:
        return self.stack1[-1]
    def empty(self) -> bool:
        return len(self.stack1) == 0
# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()

# 496. Next Greater Element I
class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        stack=[]
        next_greater={}
        for i in reversed(nums2): # Traverse from right to left
            while stack and stack[-1]<i: # Remove smaller elements – they can't be next greater
                stack.pop()
            if not stack:
                next_greater[i]=-1
            else:
                next_greater[i]=stack[-1]
            stack.append(i)
        return [next_greater[x] for x in nums1] # Build answer for nums1
        # ans=[] 
        # for k in nums1:
        #     ans.append(next_greater[k])
        # return ans

# 20. Valid Parentheses
class Solution:
    def isValid(self, s: str) -> bool:
        stack=[]
        for char in s:
            if char=='('  or char=='{' or char=='[': # opening
                stack.append(char)
            else: # closing
                if not stack:  # closing > opening
                    return False
                top=stack[-1]
                if top=='(' and char==')' or top=='{' and char=='}' or top=='[' and char==']':
                    stack.pop()
                else: # no matching
                    return False
        return not stack # opening > closing
# or
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        mapping = {')': '(', ']': '[', '}': '{'}
        
        for char in s:
            if char in mapping:
                top_element = stack.pop() if stack else '#'
                if mapping[char] != top_element:
                    return False
            else:
                stack.append(char)
        
        return not stack  # Returns True if stack is empty (valid), False otherwise.

# 387. First Unique Character in a String
from collections import defaultdict, deque
class Solution:
    def firstUniqChar(self, s: str) -> int:
        m=defaultdict(int)
        q=deque()
        for i in range(len(s)):
            if s[i] not in m:
                q.append(i)
            m[s[i]]+=1
        while q and m[s[q[0]]]>1:
            q.popleft()
        if q:
            return q[0]
        else:
            return -1

# Reverse first K of a Queue (gfg)
class Solution:
    def reverseFirstK(self, q, k):
        #code here 
        if k>len(q) or k<=0:
            return q 
        stack=[]
        for i in range(k):
            stack.append(q.popleft())
        while stack:
            q.append(stack.pop())
        rem = len(q)-k
        for _ in range(rem):
            q.append(q.popleft())
        return q 
    
# 2073. Time Needed to Buy Tickets
class Solution:
    def timeRequiredToBuy(self, tickets: List[int], k: int) -> int:
        t = 0
        for i in range(len(tickets)):  # iterate through each person
            if i <= k:  # people before or at k buy until person k finishes
                t += min(tickets[i], tickets[k])
            else:  # people after k buy only while person k still needs >1 ticket
                t += min(tickets[i], tickets[k] - 1)
        return t

# 503. Next Greater Element II
class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        stack=[]; n=len(nums); ans=[-1]*n
        for i in range(2*n-1, -1, -1):  # circular scan
            idx=i%n  # actual index
            while stack and nums[stack[-1]]<=nums[idx]: stack.pop()  # remove smaller
            ans[idx]=nums[stack[-1]] if stack else -1  # next greater
            stack.append(idx)  # push current
        return ans

# previous smaller element
arr=[3,1,0,8,6]
stack=[]; n=len(arr); ans=[-1]*n
for i in range(n):
    while stack and arr[stack[-1]]>=arr[i]: stack.pop()
    ans[i]=arr[stack[-1]] if stack else -1
    stack.append(i)
print(ans)

# 1856. Maximum Subarray Min-Product
class Solution:
    def maxSumMinProduct(self, nums: List[int]) -> int:
        def NSEL(nums):
            st=[]
            res=[-1]*n
            for i in range(n):
                while st and nums[st[-1]]>=nums[i]:
                    st.pop()
                res[i]= st[-1] if st else -1
                st.append(i)
            return res
        def NSER(nums):
            st=[]
            res=[n]*n
            for i in range(n-1,-1,-1):
                while st and nums[st[-1]]>=nums[i]:
                    st.pop()
                res[i]= st[-1] if st else n
                st.append(i)
            return res
        n=len(nums)
        pSum=[0]*(n+1)
        for i in range(n):
            pSum[i+1]=pSum[i]+nums[i]
        smallerOnLeft=NSEL(nums)
        smallerOnRight=NSER(nums)
        maxv=float('-inf')
        for i in range(n):
            minv=nums[i]
            ls=smallerOnLeft[i]
            rs=smallerOnRight[i]
            sumv=pSum[rs]-pSum[ls+1]
            minProduct=minv*sumv
            maxv=max(minProduct,maxv)
        return maxv%1000000007
    
# celebrity problem
arr=[[0,1,0],[0,0,0],[0,1,0]]
st=[]
n=len(arr)
for i in range(n):
    st.append(i)
while (len(st)>1):
    a=st.pop()
    b=st.pop()
    if arr[a][b]==1: # a knows b
        st.append(b)
    else:
        st.append(a)
celeb=st.pop()
for i in range(n):
    if i!=celeb and (arr[celeb][i]==1 or arr[i][celeb]==0):
        print("No celebrity")
        break
else:
    print("Celebrity is", celeb)

# 155. Min Stack
class MinStack:
    def __init__(self):
        self.s=[]
        self.minVal=float('inf')
    def push(self, val: int) -> None:
        if not self.s:
            self.s.append(val)
            self.minVal=val
        elif val<self.minVal:
            self.s.append((2*val)-self.minVal)
            self.minVal=val
        else:
            self.s.append(val)
    def pop(self) -> None:
        if self.s[-1]<self.minVal:
            self.minVal=(2*self.minVal)-self.s[-1]
        self.s.pop()        
    def top(self) -> int:
        if self.s[-1]<self.minVal:
            return self.minVal
        else:
            return self.s[-1]
    def getMin(self) -> int:
        return self.minVal
# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()

# 134. Gas Station
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        start=0                     # candidate start index
        currGas=0                   # current net gas
        for i in range(len(gas)):
            currGas+=gas[i]-cost[i] # update net gas
            if currGas<0:           # can't start from current segment
                currGas=0
                start=i+1           # shift start
        return -1 if sum(gas)<sum(cost) else start  # check overall feasibility (totalgas<totalcost)

# 994. Rotting Oranges
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        n,m=len(grid),len(grid[0])
        vis=[[False]*m for _ in range(n)]
        q=deque();ans=0
        for i in range(n):
            for j in range(m):
                if grid[i][j]==2:
                    q.append((i,j,0))
                    vis[i][j]=True
        # BFS
        while q:
            i,j,t=q.popleft()
            ans=max(ans,t)
            if i-1>=0 and not vis[i-1][j] and grid[i-1][j]==1:
                q.append((i-1,j,t+1))
                vis[i-1][j]=True
            if j+1<m and not vis[i][j+1] and grid[i][j+1]==1:
                q.append((i,j+1,t+1))
                vis[i][j+1]=True
            if i+1<n and not vis[i+1][j] and grid[i+1][j]==1:
                q.append((i+1,j,t+1))
                vis[i+1][j]=True
            if j-1>=0 and not vis[i][j-1] and grid[i][j-1]==1:
                q.append((i,j-1,t+1))
                vis[i][j-1]=True
        for i in range(n):
            for j in range(m):
                if grid[i][j]==1 and not vis[i][j]:
                    return -1
        return ans
        
# Stock Span Problem
def stockSpan(price,n):
    s=[]                 # stack of indices
    ans=[0]*n
    for i in range(n):
        while s and price[s[-1]]<=price[i]:  # pop smaller/equal
            s.pop()
        ans[i]=i+1 if not s else i-s[-1]     # span length
        s.append(i)                          # push index
    return ans
n=int(input())
price=list(map(int,input().split()))
res=stockSpan(price,n)
print(*res)  

# 84. Largest Rectangle in Histogram
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        n=len(heights)
        ln=[0]*n
        rn=[0]*n
        stack=[]
        # right nearest shortest height for each height
        for i in range(n-1,-1,-1):
            while stack and heights[stack[-1]]>=heights[i]:
                stack.pop()
            rn[i]= n if not stack else stack[-1]  # if no right nearest short height take n instead of -1 because eg. for 2 -> r-l-1 -> (-1)-1-1=-3 incorrect so, (6-1-1)=4 which gives correct . [2,1,5,6,2,3]
            stack.append(i) 
        stack.clear()
        # left nearest shortest for each height
        for i in range(n):
            while stack and heights[stack[-1]]>=heights[i]:
                stack.pop()
            ln[i]= -1 if not stack else stack[-1]
            stack.append(i)
        maxarea=0 
        for i in range(n):
            width=rn[i]-ln[i]-1
            curarea=heights[i]*width
            maxarea=max(maxarea,curarea)
        return maxarea