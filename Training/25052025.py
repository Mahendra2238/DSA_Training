# s="abcdecfbgce"
# m=0
# l=0
# d={}
# for r in range(len(s)):
#     if s[r] in d and d[s[r]]>=l:
#         l=d[s[r]]+1     
#     d[s[r]]=r
#     m=max(m,r-l+1)
# print(m)

# find the longest non repeating substring which has m and n values
# s="abcedachfaebghd"
# ml=0
# l=0
# si=0
# d={}
# m='c'
# n='d'
# for r in range(len(s)):
#     if s[r] in d and d[s[r]]>=l:
#         l=d[s[r]]+1
#     d[s[r]]=r
#     if r-l+1>ml and m in d and n in d and d[m]>=l and d[n]>=l:
#         ml=r-l+1
#         si=l
# print(ml)
# print(s[si:si+ml])

# have a deck of cards arr you can remove k cards first or last card from deck such that total score value is max  
# dc=[4,2,7,20,8,6,8,1]  
# k=3
# s=sum(dc[:k])
# l=k-1
# r=len(dc)-1
# m=s
# while l>=0 :
#     s-=dc[l]
#     s+=dc[r]
#     r-=1
#     l-=1
#     m=max(m,s)
# print(m)

# dc=[4,3,2,5,6,1,12,3]  
# k=4
# n=len(dc)
# sl=0
# for i in range(k):
#     sl+=dc[i]
# m=sl
# sr=0
# for i in range(k-1,-1,-1):
#     sl=sl-dc[i]
#     sr=sr+dc[n-1]
#     m=max(m,sl+sr)
#     n-=1
# print(m)

# 1004. Max Consecutive Ones III
# a=[1,1,1,0,0,0,1,1,1,1,0]
# k=2
# l=z=m=0
# for r in range(len(a)):
#     if a[r]==0:
#         z+=1
#     if z>k:
#         if a[l]==0:
#             z=z-1
#         l=l+1
#     if z<=k:
#         m=max(m,r-l+1)
# print(m)

# from collections import defaultdict
# fruits=[1,2,3,2,2]
# l=m=0
# b=defaultdict(int) #{}
# for r in range(len(fruits)):
#     # if b[fruits[r]] not in b:
#     #     b[fruits[r]]=1
#     # else:
#     b[fruits[r]]+=1
#     while len(b)>2:
#         b[fruits[l]]-=1
#         if b[fruits[l]]==0:
#             del b[fruits[l]]
#         l+=1
#     m=max(m,r-l+1)
# print(m)

# max no of doctors required 
# s=[920,945,1110,1230,1235,1245,1340,1700]
# e=[930,1130,1120,1250,1250,1415,1400,1730]
# md=0
# l=e[0]
# for i in range(len(s)):
#     if s[i]>l:

from collections import defaultdict
d=defaultdict(int)
d[1]=2
d[2]=1
d[4]=3
print(len(d))       
