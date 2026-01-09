# given an arr with unique and duplicate ele arr=[2,4,1,5,8,4,7].
# def maxProf(arr):
#     maxp=0
#     for i in range(len(arr)):
#         for j in range(i+1,len(arr)):
#             prof=arr[j]-arr[i]
#             if prof>maxp:
#                 maxp=prof
#     return maxp
# arr=[2,4,1,5,8,4,7]
# print(maxProf(arr))

# arr=[2,4,1,5,8,4,7]
# minp=arr[0]
# maxp=0
# for p in arr:
#     if p<minp:
#         minp=p
#     if (p-minp)>maxp:
#         maxp=p-minp
# print(maxp)

# distinct island problem

# a=[[0,1,1,0,1],
#    [1,1,0,0,1],
#    [0,0,0,1,1],
#    [0,1,0,0,0]]
# l=[8,7,6,5,2]
# ans=[]
# for i in range(len(a)):
#     sume=0
#     for j in range(len(a[0])):
#         if a[i][j]==1:
#             sume+=l[j]
#     ans.append(sume)
# print(ans)

# def ratrace(a, n, r, c):
#     if r >= n or c >= n or a[r][c] == 0: #r < 0 or c < 0
#         return 0
#     if r == n - 1 and c == n - 1:
#         return 1
#     return ratrace(a, n, r, c + 1) + ratrace(a, n, r + 1, c)  # right + down
# a = [[1, 0, 0, 0],
#      [1, 1, 1, 1],
#      [1, 0, 0, 1],
#      [1, 1, 1, 1]]
# print(ratrace(a, len(a), 0, 0))

# n=int(input())
# a=[]
# for i in range(n):
#     row=list(map(int,input().split(" ")))
#     a.append(row)
# print(a)
# print(ratrace(a,len(a),0,0,0))

# def land(a,i,j,n):
#     if i==n or j==n or i<0 or j<0 or a[i][j]==0 or a[i][j]==2:
#         return 0
#     if a[i][j]==1:
#         a[i][j]=2
#     land(a,i-1,j,n)
#     land(a,i,j-1,n)
#     land(a,i+1,j,n)
#     land(a,i,j+1,n)
# a=[[1,0,0,1,1],[1,0,0,0,1],[0,0,0,1,0],[1,0,0,0,0],[1,0,0,0,1]]
# c=0
# for i in range(5):
#     for j in range(5):
#         if (a[i][j]==1):
#             land(a,i,j,5)
#             c=c+1
# print(c)

# no of unburnt trees
# def unburnt(a,n,i,j):
#     if i==n or j==n or i<0 or j<0 or a[i][j]==0 or a[i][j]==2:
#         return 0
#     if a[i][j]==1:
#         a[i][j]=2
#     unburnt(a,n,i-1,j)
#     unburnt(a,n,i,j-1)
#     unburnt(a,n,i+1,j)
#     unburnt(a,n,i,j+1)
# a=[[1,0,0,1,1,1],[1,1,1,0,0,0],[0,0,1,1,1,1],[1,1,1,0,0,0],[0,0,0,0,0,1],[1,0,0,1,0,0]]
# c=0
# unburnt(a,6,0,0 )
# for i in range(6):
#     for j in range(6):
#         if (a[i][j]==1):
#             c=c+1
# print(c)

# frog possible no of paths without into traps only right and down
# def paths(n,x,y,trap):
#     if x<=0 or y<=0 or x>n or y>n or (x,y) in trap:
#         return 0
#     if x==n and y==n:
#         return 1
#     return paths(n,x+1,y,trap) + paths(n,x,y+1,trap)
# n=5  #int(input())
# x,y=2,3  #map(int(input().split()))
# trap=[(2,1),(4,1),(5,2),(3,5)]

# import math
# re=[]
# def allbin(n,s=''):
#     if(len(s)==n):
#         re.append(s)
#         return
#     allbin(n,s+'0')
#     allbin(n,s+'1')
# n=int(input())
# m=int(math.log(n,2)+1)
# allbin(m)
# for i in range(n+1):
#     print(re[i])
#  when length becomes log n we stop

# import math
# def allbinary(n,s=''):
#     global a
#     if a==-1:
#         return a
#     if(len(s)==n):
#         print(s)
#         a=a-1
#         return 
#     allbinary(n,s+'0')
#     allbinary(n,s+'1')
# a=5
# n=int(math.log(a,2))+1
# allbinary(n)

# count possible paranthesis (if length is 2n return)
