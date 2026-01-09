# l=[[2,3,4],[9,8,7],[1,0,6]]
# k=1 
# f=0
# for i in range(len(l)):
#     for j in range(len(l[0])):
#         if l[i][j]==k:
#             print(f"{l[i][j]} found at index {i},{j}")
#             f=1
#             break
#     if f==1:
#         print("found")
#         break
# else:
#     print("not found")

# search ele in sorted matrix   o(n+logm)
# def bin(s,e,r,k):
#     while s<=e:
#         m=(s+e)//2
#         if r[m]==k:
#             print(r[m])
#             return True
#         if k<r[m]:
#             e=m-1
#         else:
#             s=m+1
#     return False
# def search(a,k):
#     for r in a:
#         if r[0]<=k<=r[-1]:
#             if bin(0,len(r)-1,r,k):
#                 print("Found")
#                 break
#     else:
#         print("Not found")
# a=[[2,3,7,8],[9,15,20,22],[23,26,35,37],[38,39,42,43]]
# k=23
# search(a,k)

# o(logn+logm)
# def search(a,k):
#     n,m=len(a),len(a[0])
#     l=0
#     r=(n*m)-1
#     while l<=r:
#         mid=(l+r)//2
#         if a[mid//m][mid%m]==k:
#             return [int(mid/m),mid%m]
#         if a[mid//m][mid%m]>k: 
#             r=mid-1
#         else:
#             l=mid+1
#     return [-1,-1]      
# a=[[1,2,3,7,8],[9,15,20,21,22],[23,25,26,35,37],[38,39,42,43,45]]
# k=37
# print(search(a,k))

# 1011. Capacity To Ship Packages Within D Days        
# possible or not 
# w=[1,2,3,4,5,6,7,8,9,10]
# c=12
# sh=0
# d=[]
# for i in range(len(w)):
#     if w[i]>c:
#         d.append(sh) 
#         c=12
#         sh=0
#     sh+=w[i]
#     c-=w[i]
# if sh>c:
#     d.append(w[i])
# print(d)
# print(len(d))

# using binary
# def canShip(weights,days,capacity):
#     d=1
#     curr=0
#     for w in weights:
#         if curr+w>capacity:
#             d+=1
#             curr=0
#         curr+=w
#     return d<=days
# def cap(w,d):
#     l,r=max(w),sum(w)
#     while l<r:
#         m=(l+r)//2
#         if canShip(w,d,m):
#             r=m
#         else:
#             l=m+1
#     return l
# w=[1,2,3,4,5,6,7,8,9,10]
# d=5    
# print(cap(w,d))

# when rows and col are sorted 
# def searchMatrix(a,k):
#     r = 0
#     c = len(a[0]) - 1
#     while r < len(a) and c >= 0:
#         if a[r][c] == k:
#             return [a[r][c],r,c]#True
#         elif a[r][c] < k:
#             r += 1
#         else:
#             if a[r][c]<k:
#                 r+=1
#             c -= 1
#     return False
# a=[[2,4,6,8],[5,7,9,10],[8,12,13,15],[20,23,26,28],[30,36,37,45]]
# k = 45
# print(searchMatrix(a,k))

# 2sum
# a=[2,3,6,7,11,15]
# k=9
# i,j=0,len(a)-1
# d=[]
# while i<j:
#     s=a[i]+a[j]
#     if s==k:
#         d.append([a[i],a[j]])
#         i+=1
#         j-=1
#     elif s<k:
#         i+=1
#     else:
#         j-=1
# print(d)

# subset king 1000 wines 1 is poisoned it will effect in 10 hrs and party is also within 10 hrs how many min soldiers required to test
# answer= 10    1000 possibilities  2^10=1024
# Each wine can be represented as a 10-bit binary number (2¹⁰ = 1024 > 1000).
# So, assign each bit to a soldier. A soldier drinks from wines where their bit is set to 1.
# Each soldier represents a bit position from 0 to 9. After 10 hours:
# Soldiers who die form a binary pattern.
# That binary number = index of poisoned wine.

# 3sum
# a = [-1, 0, 1, 2, -1, -4]
# a.sort()
# t = 0
# d = []
# for i in range(len(a) - 2):
#     if i > 0 and a[i] == a[i - 1]:
#         continue
#     j = i + 1
#     k = len(a) - 1
#     while j < k:
#         s = a[i] + a[j] + a[k]
#         if s == t:
#             d.append([a[i], a[j], a[k]])
#             j += 1
#             k -= 1
#             while j < k and a[j] == a[j - 1]:
#                 j += 1
#             while j < k and a[k] == a[k + 1]:
#                 k -= 1
#         elif s < t:
#             j += 1
#         else:
#             k -= 1
# print(d)
