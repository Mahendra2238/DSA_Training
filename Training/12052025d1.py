# import math
# def is_prime(x):
#     isPrime=True
#     if n<2:
#         isPrime=False
#     for i in range(2,int(math.sqrt(n)+1)): #int(n ** 0.5) + 1
#         if n%i==0:
#             isPrime=False
#             break
#     if isPrime:
#         if n>200:
#             print("Is prime and Greater than 200")
#         else:
#             print("Is prime and Less than or equal to 200")
#     else:
#         print("Not a prime")
# n=int(input("enter a number: "))
# is_prime(n)

# def remove_dup(lst):
#     seen=set()
#     unique=[]
#     for i in lst:
#         if i not in seen:
#             unique.append(i)
#             seen.add(i)
#     return unique
# lst=[8,2,3,4,3,3,4,5,6,7,9,2]
# print(remove_dup(lst))    

# a=7
# a=a-2
# b=0
# b=b+1+1
# b=b-1
# c=a^b
# print(c)

# n=int(input())
# sum=n*(n+1)//2
# # for i in range(1,n):
# #     print(i,end="+")
# # print(n,end="=")
# print(sum)

'''binary search = O(log n)
    n/2/2/2...(divide & conquer)
    n/2^k = 1
    n=2^k
    log n = log 2^k
    log n = k log 2
    log n = k
'''

# lst=[2,3,4,3,2,5,5]
# for i in range(len(lst)):
#     c=0
#     for j in range(len(lst)):
#         if (lst[i]==lst[j] and i!=j):
#             c=1
#     if c==0:
#         print(lst[i])
#         break
 
# from collections import Counter
# lst = [2, 3, 4, 3, 2, 5, 5]
# freq = Counter(lst)
# for num in lst:
#     if freq[num] == 1:
#         print(num)
#         break

# lst = [2, 3, 4, 3, 2, 5, 5]
# freq = {}
# # Count the frequency of each element
# for num in lst:
#     freq[num] = freq.get(num, 0) + 1
# # Find the first element with count 1
# for num in lst:
#     if freq[num] == 1:
#         print(num)
#         break

# sorted arr inp using sliding window 
# O(n)
# a=[2,2,4,4,6,6,7,7,8,8,9]
# res=0
# for num in a:
#     res^=num
# print(res)

# o(n)
# a=[2,2,4,4,6,6,7,7,8,8,9]
# for i in range(0,len(a)-1,2):
#     if a[i]!=a[i+1]:
#         print(a[i])
#         break
# else:
#     print(a[-1])

# o(logn)
# a=[2,2,4,4,6,6,7,7,8,8,9]
# l,r=0,len(a)-1
# while l<r:
#     mid=l+(r-l)//2
#     if mid%2==1:
#         mid-=1
#     if a[mid]==a[mid+1]:
#         l=mid+2
#     else:
#         r=mid
# print(a[l])
