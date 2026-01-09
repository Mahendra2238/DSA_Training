# given and array and k value . print the frequency of k in array  [2,4,6,3,3,2,6,1,2,3,6,6,9]
# def freq(a,k,i,c):
#     if i>=len(a):
#         return c
#     if a[i]==k:
#         return freq(a,k,i+1,c+1)
    
#     return freq(a,k,i+1,c)
# a=[2,4,6,3,3,2,6,1,2,3,6,6,9]
# k=int(input())
# print(freq(a,k,0,0))

# if a number in arr has k(freq)
# a=[2,4,6,3,3,2,6,1,2,3,6,6,9]
# k=1
# dic={}
# for i in range(len(a)):
#     c=0
#     for j in range(len(a)):
#         if a[i]==a[j]:
#             c+=1
#     dic[a[i]]=c
# for key,val in dic.items():
#     if val==k:
#         print(key)
    
# def freqm(a,k,dic,i=0,m=0,c=0):
#     if m >= len(a):
#         return None
#     if i >= len(a):
#         dic[a[m]] = c
#         if dic[a[m]] == k:
#             return a[m]
#         return freqm(a, k, dic, 0, m + 1, 0)
#     if a[i]==a[m]:
#         return freqm(a,k,dic,i+1,m,c+1)
#     return freqm(a,k,dic,i+1,m,c)
# a=[2,4,6,3,3,2,6,1,2,3,6,6,9]
# k=4
# dic={}
# print(freqm(a,k,dic))

# without dict
# def freq(a, x, i=0):
#     if i == len(a):
#         return 0
#     if a[i] == x:
#         return 1 + freq(a, x, i + 1)
#     return freq(a, x, i + 1)
# def iterate(x, f, i=0):
#     if i == len(x):
#         return "not found"
#     if freq(x, x[i]) == f:
#         return x[i]
#     return iterate(x, f, i + 1)
# a = [2, 4, 6, 3, 3, 2, 6, 1, 2, 3, 6, 6, 9]
# k = 4
# print(iterate(a, k))

# # with dict
# def value(x,f,d,i=0):
#     if i==len(x):
#         return "not found"
#     if d[x[i]]==f:
#         return x[i]
#     return value(x,f,d,i+1)
# def feq_d(x,f,d={},i=0):
#     if i==len(x):
#         return value(list(d.keys()),f,d)
#     if x[i] not in d:
#         d[x[i]]=1
#     else:
#         d[x[i]]+=1
#     return feq_d(x, f, d, i + 1)
# a = [2, 4, 6, 3, 3, 2, 6, 1, 2, 3, 6, 6, 9]
# f = 4
# print(feq_d(a,f))  

# subset and subsequence:
# def subsets(a):
#     r=[]
#     def sb(i,p):
#         if i==len(a):
#             r.append(p)
#             return
#         sb(i+1,p)
#         sb(i+1,p+[a[i]])
#     sb(0,[])
#     return r
# a=[2,3,4]
# print(subsets(a))

# def subset(x,s=[],i=0):
#     if i==len(x):
#         print(s)
#         return
#     subset(x,s+[x[i]],i+1)
#     subset(x,s,i+1)
# a=[2,3,4]
# subset(a)

# check k= sum of any subset
# def subset_sum(x,k,i=0):
#     if k==0:
#         return True
#     if i==len(x) or k<0:
#         return False
#     if subset_sum(x,k-x[i],i+1):
#         return True
#     if subset_sum(x,k,i+1):
#         return True
#     return False
# a=[2,4,6]
# k=5
# print(subset_sum(a,k))

# def subset_sum(x,k,i):
#     if k==0:
#         return True
#     if i==0:
#         return False
#     if x[i-1]>k:
#         subset_sum(x,k,i-1)
#     return subset_sum(x,k-x[i],i-1)
#     # return subset_sum(x,k,i+1) or subset_sum(x,k-x[i],i+1)
# a=[2,3,4]
# k=6
# print(subset_sum(a,k,len(a)))

# def subset(x,k,s=[],i=0):
#     if i==len(x):
#         if k==0:
#             print(s)
#         return
#     if x[i]<=k:
#         subset(x,k-x[i],s+[x[i]],i+1)
#     subset(x,k,s,i+1)
# a=[2,3,4,5]
# k=9
# subset(a,k)

#  def mincoins(c,k,s=[],i=0,minc=float('inf')):
#     if i==len(c):
#         if k==0:
#             return len(s)
#         return minc
#     if c[i]<=k:
#         minc=min(minc,mincoins(c,k-c[i],s+[c[i]],i+1,minc))
#     minc=min(minc,mincoins(c,k,s,i+1,minc))
#     return minc
# c=[2,4,6,8]
# k=14
# res=mincoins(c,k)
# print(res if res != float('inf') else -1) 

# def subset(x,k,s=0,i=0,m=9999999):
#     if i==len(x):
#         if k==0: 
#             if s<m:
#                 m=s
#         return m
#     if x[i]<=k:
#         m=subset(x,k-x[i],s+1,i+1,m)
#     m=subset(x,k,s,i+1,m)
#     return m
# a=[2,4,6,8]
# k=14
# print(subset(a,k))
# largest even no and smallest odd num

# lst=list(map(int,input().split(" ")))
# leven,sodd=0,99999
# for i in range(len(lst)):
#     if lst[i]%2==0:
#         if lst[i]>leven:
#             leven=lst[i]
#         # leven=max(leven,lst[i])
#     else:
#         if lst[i]<sodd:
#             sodd=lst[i]
#         # sodd=min(sodd,lst[i])
# print("Largest even: ",leven,"\nSmallest odd: ",sodd)

lst=list(map(int,input().split(" ")))
fl=0
sl=0
for i in lst:
    if i>fl:
        sl,fl=fl,i
    elif i>sl and i!=fl:
        sl=i
print(sl)
