# Dynamic Programming
# recursion  - doing so much unwanted work
# def fibo(n):
#     if n==0:
#         print("hi")
#         return 0
#     if n==1:
#         print("hi")
#         return 1
#     return fibo(n-1)+fibo(n-2)
# print(fibo(7)) #21 times hi

# dp memorization and tabulation
# def fibo(n):
#     # print("hi") 13 times
#     if dp[n-1]!=-1:
#         return dp[n-1]
#     if n==0:
#         print("hi")
#         return 0
#     if n==1:
#         print("hi")
#         return 1
#     dp[n-1]=fibo(n-1)+fibo(n-2)
#     return dp[n-1] #3 times hi
# n=7
# dp=[-1]*n
# # dp[0]=1
# # dp[1]=1
# print(fibo(n))
# print(dp)

# n=7
# f1=1
# f2=1
# for i in range(2,n):
#     a=f1+f2
#     f1=f2
#     f2=a
# print(f2)

# a frog can jump either 1 or 2 jumps find min  
# memorization
# def jumps(n):
#     if n<=1:
#         return dp[n]
#     one=jumps(n-1)+abs(jh[n]-jh[n-1])
#     two= jumps(n-2)+abs(jh[n]-jh[n-2])
#     dp[n]=min(one,two)
#     return dp[n]
# jh=[10,20,30,10]
# dp=[0]*len(jh)
# dp[1]=abs(jh[1]-jh[0])
# print(jumps(len(jh)-1))
# print(dp)

# tabulation
# h=[10,20,30,10,30,20,10]
# dp=[0]*len(h)
# dp[1]=abs(h[1]-h[0])
# for i in range(2,len(h)):
#     dp[i]=min((dp[i-1]+abs(h[i]-h[i-1]),(dp[i-2]+abs(h[i]-h[i-2]))))
# print(dp[len(h)-1])
# print(dp)

# k jumps
# h=[10,20,30,10,30,20,10]
# k=3
# dp=[9999]*len(h)
# dp[1]=abs(h[1]-h[0])
# for i in range(2,len(h)):
#     m=float('inf')
#     for j in range(1,k+1):
#         m=min(m,(dp[i-j]+abs(h[i]-h[i-j])))
#     dp[i]=m
# print(dp[len(h)-1])
# print(dp)

# given times(s,e) and incomes respectively find max income can be earned doing part time jobs
# def mi():
#     for i in range(len(t)):
#         # dp[i]=ic[i]
#         for j in range(0,i):
#             if t[j][1]<=t[i][0]:
#                 dp[i]=max(dp[i],dp[j]+ic[i])
#     print(max(dp))
#     print(dp)
# t=[(1,3),(2,5),(4,6),(6,7),(5,8),(7,9)]
# ic=[5,6,5,4,11,2]
# dp=ic.copy() #[0]*len(ic)
# mi()


