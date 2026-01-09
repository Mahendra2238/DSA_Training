# count possible paranthesis (if length is 2n return)
# def para(n, oc, cc, s=''):
#     if len(s) == 2 * n:
#         print(s)
#         return 1
#     c = 0
#     if oc < n:
#         c += para(n, oc + 1, cc, s + '(')
#     if cc < oc:
#         c += para(n, oc, cc + 1, s + ')')
#     return c
# n = int(input())
# print(para(n, 0, 0))

# Bubble sort o(n^)
# def bb(a,n):
#     for i in range(n-1):
#         for j in range(n-1-i):
#             if a[j]>a[j+1]:
#                 a[j],a[j+1]=a[j+1],a[j]
#     return a
# a=[5,1,4,2,3,7]
# print(bb(a,len(a)))

# efficient bubble sort   o(n^2) best case o(n) space o(1)
# def bb(a,n):
#     c=0
#     for i in range(n-1):
#         swapped=False
#         for j in range(n-1-i):
#             c+=1
#             if a[j]>a[j+1]:
#                 a[j],a[j+1]=a[j+1],a[j]
#                 swapped=True
#         if not swapped:
#             break 
#         print(a)
#     print(c)
# a=[3,5,1,2,4,7,8]
# bb(a,len(a))

# kth largest element in bubble sort

# a=[3,5,1,2,4,7,7,8]
# k=3
# a=list(set(a))
# for i in range(len(a)-1):
#     swapped=False
#     for j in range(len(a)-1-i):
#         if a[j]>a[j+1]:
#             a[j],a[j+1]=a[j+1],a[j]
#         swapped=True
#     if not swapped:
#         break 
#     if i==k:
#         print(a[len(a)-i])
# print(a)

# a=['c','e','a','b','f']
# for i in range(len(a)-1):
#     swapped=False
#     for j in range(len(a)-1-i):
#         if a[j]>a[j+1]:
#             a[j],a[j+1]=a[j+1],a[j]
#         swapped=True
#     if not swapped:
#         break 
# print(a)

# 2d list second ele comparision
# a=[[2,3],[5,1],[1,4],[2,7],[1,3]]
# for i in range(len(a)-1):
#     swapped=True
#     for j in range(len(a)-1-i):
#         if a[j][1]>a[j+1][1]:
#             a[j],a[j+1]=a[j+1],a[j]
#         swapped=False
#     if swapped:
#         break 
# print(a)

# a=["caft","giraffee","apple","cabt"]
# for i in range(len(a)-1):
#     swapped=False
#     for j in range(len(a)-1-i):
#         if a[j][0]>a[j+1][0]:
#             a[j],a[j+1]=a[j+1],a[j]
#             swapped=True
#         elif a[j][0]==a[j+1][0] and a[j][1]>a[j+1][1]:
#             a[j],a[j+1]=a[j+1],a[j]
#             swapped=True
#     if not swapped:
#         break 
# print(a)

# def isPrime(num):
#     for i in range(2,int(num//2)+1):
#         if num%i==0:
#             return False
#     return True
# p=[[4,13,12],[8,10,5],[7,9,20],[11,8,3]]
# n=len(p)
# for i in range(n-1):
#     swapped=False
#     for j in range(n-1-i):
#         for k in range(len(p[j])):
#             if isPrime(p[j][k]):
#                 r=k
#         for k in range(len(p[j+1])):
#             if isPrime(p[j+1][k]):
#                 r1=k
#         if p[j][r]>p[j+1][r1]:
#             p[j],p[j+1]=p[j+1],p[j]
#         swapped=True
#     if not swapped:
#         break 
# print(p)
# here we are finding prime index again and again 

# def isPrime(num):
#     if num<2:
#         return False
#     for i in range(2,int(num**0.5)+1):
#         if num%i==0:
#             return False
#     return True
# def primes(p):
#     pr=[]
#     for row in p:
#         for l in row:        
#             if isPrime(l):
#                 n=l
#         pr.append(n)
#     return pr
# p=[[4,13,12],[8,10,5],[7,9,20],[11,8,3]]
# n=len(p)
# pr=primes(p)
# for i in range(n-1):
#     swapped=False
#     for j in range(n-1-i):
#         if pr[j]>pr[j+1]:
#             p[j],p[j+1]=p[j+1],p[j]
#             pr[j],pr[j+1]=pr[j+1],pr[j]
#             swapped=True
#     if not swapped:
#         break 
# print(p)

#  effiecient code
# def prime(x):
#     for i in x:
#         for j in range(2,int(i**0.5)+1):
#             if(i%j==0):
#                 break
#         else:
#             return i
# a=[[4,13,12],[8,10,5],[7,9,20],[14,8,3]]
# b=[]
# for i in a:
#     b.append(prime(i))
# for i in range(len(b)-1):
#     f=0
#     for j in range(len(b)-1-i):
#         if(b[j]>b[j+1]):
#             b[j],b[j+1]=b[j+1],b[j]
#             a[j],a[j+1]=a[j+1],a[j]
#             f=1
#     if(f==0):
#         break
# print(a)

# st="an apple a day keeps doctor away"
# s=list(st.split(" "))
# n=len(s)
# for i in range(n-1):
#     for j in range(n-1-i):
#         f=0
#         if len(s[j])>len(s[j+1]):
#             s[j],s[j+1]=s[j+1],s[j]
#             f=1
#     if f==0:
#         break        
# print(' '.join(s))

# s="an apple a day keeps doctor away" #.split()
# words=s.split(" ")
# lw=[]
# for i in words:
#     lw.append(len(i))
# n=len(words)
# for i in range(n-1):
#     for j in range(n-1-i):
#         f=0
#         if lw[j]>lw[j+1]:
#             words[j],words[j+1]=words[j+1],words[j]
#             lw[j],lw[j+1]=lw[j+1],lw[j]
#             f=1
#     if f==0:
#         break 
# print(' '.join(words))

# ip = [7, 2, 6, 3, 6, 7, 7, 2, 2, 2]
# dictu = {}
# for i in ip:
#     if i not in dictu:
#         c = 0
#         for j in range(len(ip)):
#             if i == ip[j]:
#                 c += 1
#         dictu[i] = c
# n = len(ip)
# print(dictu)
# for i in range(n - 1):
#     f = 0
#     for j in range(n - 1 - i):
#         if dictu[ip[j]] > dictu[ip[j + 1]]:
#             ip[j], ip[j + 1] = ip[j + 1], ip[j]
#             f = 1
#     if f == 0:
#         break
# print(ip)

# arrange elements respect to frequrencies bucket sort 
# ip = [7, 2, 6, 3, 6, 7, 7, 2, 2, 2,8,8,9,9,9]
# d={}
# for i in ip:
#     if i in d:
#         d[i]+=1
#     else:
#         d[i]=1
# print(d)
# mf=max(d.values())
# l=[]
# for i in range(mf+1):
#     l.append([])
# print(l)
# for i in d.items():
#     print(i)
#     l[i[1]].append(i[0])
# print(l)
# c=[]
# for i in range(len(l)):
#     for j in l[i]:
#         c.extend([j]*i)
# print(c)

# a.sort() is best it default use quick sort nlogn unlike bubble o(n^@)
# a=["hello","apple","hey","giraffee"]
# a.sort()
# print(a)

# def qwe(x):
#     return x[1]
# a=[[3,9],[4,2],[10,1],[5,7],[3,8]]
# a.sort(key=qwe)
# print(a)

# def qwe(x):
#     return x[-1]
# a=[[3,9,5],[4,2,2],[10,1,5],[5,7,7],[3,8,4]]
# a.sort(key=qwe)
# print(a)

def qwe(x):
    for i in x:
        for j in range(2,int(i**0.5)+1):
            if i%j==0:
                break
            else:
                return 1
    return 0
a=[[3,9,5],[4,2,2],[10,1,5],[5,7,7],[3,8,4]]
a.sort(key=qwe,reverse=True)
print(a)


        
