# # Prime Numbers
# import math
# def isPrime(n):
#     if n<2:
#         return False
#     for i in range(2,int(math.sqrt(n))+1): # 2 to sqrt(n) # i*i<=n or i<=sqrt(n)
#         if n%i==0:
#             return False #"Non prime"
#     return True #"Prime"
# # print(isPrime(7))
# # num th prime             2 3 5 7 11 13 17 19 23 29..
# def nth_prime(num):
#     c=0
#     i=1
#     while True:
#         i+=1
#         if isPrime(i):
#             c+=1
#             if c==num:
#                 return i
# print(nth_prime(100))

# optimized
# Sieve of Eratosthenes
# count primes between 1 to 50
# def sieve(n):
#     primes=[True]*(n+1)
#     primes[0]=primes[1]=False
#     c=0
#     for i in range(2,n+1):
#         if isPrime(i):
#             c+=1
#             for j in range(i*2,n+1,i):
#                 primes[j]=False
#     return [k for k, isPrime in enumerate(primes) if isPrime]
#     return c
# def isPrime(n):
#     if n<2:
#         return False
#     for i in range(2,int(math.sqrt(n))+1):
#         if n%i==0:
#             return False 
#     return True
# print(sieve(50))

# # nth prime
# def nsieve(n):
#     primes=[True]*(n+1)
#     primes[0]=primes[1]=False
#     for i in range(2,int(n**0.5)+1):
#         if primes[i]:
#             for j in range(i*i,n+1,i):
#                 primes[j]=False
#     return [k for k, isP in enumerate(primes) if isP]
# def nth_prime_in_range(n,limit=10**4):
#     primearr=nsieve(limit)
#     if n<=len(primearr):
#         return primearr[n-1] # 0-indexed
#     return None # not enough primes within the range
# print(nth_prime_in_range(100))

# Digits in a Number
# print ((int)(math.log10(3568)+1))
# or
# n=3568
# c=0
# while n!=0:
#     d=n%10
#     print(d)
#     n//=10
#     c+=1
# print("count- ",c)

# Armstrong Number
# n=153 #371
# nc=n
# soc=0
# while n!=0:
#     d=n%10
#     soc+=(d**3)
#     n//=10
# if nc==soc:
#     print("Armstrong")
# else:
#     print("Not a Armstrong")

# GCD (greatest common divisor or highest common factor)   
# 20 -> (2x2)x5x(1)
# 28 -> (2x2)x7x(1)
# gcd=4
# a=20
# b=28
# if a==0:
#     gcd=b
# elif b==0:
#     gcd=a
# else:
#     gcd=1
#     for i in range(1,min(a,b)):
#         if a%i==0 and b%i==0:
#             gcd=i
# print(gcd).

# or
# Euclid's Algorithm
# gcd(a,b)= gcd(a-b,b),a>b
#           gcd(a,b-a),b>a
# gcd(20,28)->[gcd(20,8)->gcd(12,8)->gcd(4,8)]->gcd(4,4)->gcd(0,4)
# to optimize repeated subtractions we can do modulo
# gcd(20,8)->gcd(20,8)->gcd(4,8)->gcd(4,0)
# gcd(a,b)= gcd(a%b,b),a>b
#           gcd(a,b%a),b>a
# def gcd(a,b):
#     while a>0 and b>0:
#         if a>b:
#             a=a%b
#         else:
#             b=b%a
#     return b if a==0 else a
# print(gcd(20,28))

# or recursion
# def gcdRec(a,b):
#     if b==0:
#         return a
#     return gcdRec(b,a%b)
# print(gcdRec(20,28))

# LCM(least/loswest common multiple)
#  axb=gcd(a,b)xlcm(a,b)
# lcm=(axb)/(gcd(a,b))
# def lcm(a,b):
#     return (a*b)//gcdRec(a,b)
# print(lcm(20,28))

# Reverse a Number
# INT_MIN=-2**31    # -2147483648
# INT_MAX=(2**31)-1 # 2147483647
# def rev(n):
#     revN=0
#     while n!=0:
#         d=n%10
#         if revN>INT_MAX//10 or revN<INT_MIN//10: #if constraint given overflow check
#             return 0
#         revN=(revN*10)+d
#         n//=10
#     return revN
# print(rev(4537))

# # Palindrome Number
# def palindrome(n):
#     if n<0:
#         return False
#     if rev(n)==n:
#         return "Palindrome"
#     return "Not Palindrome"
# print(palindrome(131))

# Modulo Arithmetics
# ans -> high 
# ans%(10^7+9)
# x%n -> [0,n-1]
# 100%3 -> [0,1,2]
# n=50 -> 50! o(n!)
# properties:
# (x+y)%m = x%m + y%m
# (x-y)%m = x%m - y%m
# (x*y)%m = x%m * y%m
# ((x%m)%m)%m = x%m
# 100%3=1
# ((100%3)%3)%3 =1

# x=8, y=9, m=3
# (8+9)%3=17%3=3
# 8%3+9%3=3+0=3