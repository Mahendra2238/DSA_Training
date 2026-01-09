# coin change problem
# coin=[2,3,4,5]
# amt=12
# # if amt>sum(coin):
# #     return "Not possible"
# dp=[[0]*(amt+1) for i in range(len(coin))]
# for i in range(len(coin)):
#     dp[i][0]=1
# dp[0][coin[0]]=1
# for i in range(1,len(coin)):
#     for j in range(1,amt+1):
#         if j<coin[i]:
#             dp[i][j]=dp[i-1][j]
#         else:
#             dp[i][j]=dp[i-1][j] or dp[i-1][j-coin[i]]
#         # elif dp[i-1][j]==1:
#         #     dp[i][j]=dp[i-1][j]
#         # else:
#         #     if dp[i-1][j-coin[i]]==1:
#         #         dp[i][j]=1
# print(dp)
# if dp[-1][-1]==1:
#     print("True")
# else:
#     print("False")

# Min no of coins required to form sum
# coin=[2,3,4,5]
# k=12
# dp=[[0]*(k+1) for i in range(len(coin))]
# dp[0][coin[0]]=1
# for i in range(1,len(coin)):
#     for j in range(k+1):
#         dp[i][coin[i]]=1
#         if j<coin[i]:
#             dp[i][j]=dp[i-1][j]
#         else:
#             if dp[i-1][j-coin[i]]!=0:
#                 if dp[i-1][j]!=0:
#                     dp[i][j]=min(dp[i-1][j],1+dp[i-1][j-coin[i]])
#                 else:
#                     dp[i][j]=dp[i-1][j-coin[i]]+1
#             else:
#                 dp[i][j]=dp[i-1][j]
# for i in dp:
#     print(i)
# if dp[-1][-1]!=0:
#     print("min coins: ",dp[-1][-1])
# else:
#     print("No")

#sr
# coin=[2,3,4,5]
# k=12
# dp=[[0]*(k+1) for i in range(len(coin))]
# dp[0][coin[0]]=1
# for i in range(1,len(coin)):
#     for j in range(k+1):
#         dp[i][coin[i]]=1
#         if j<coin[i]:
#             dp[i][j]=dp[i-1][j]
#         elif coin[i]==j:
#             dp[i][j]=1
#         else:
#             if dp[i-1][j-coin[i]]!=0 and dp[i-1][j]!=0:
#                 dp[i][j]=min(dp[i-1][j],1+dp[i-1][j-coin[i]])
#             elif dp[i-1][j]!=0:
#                 dp[i][j]=dp[i-1][j]
#             elif dp[i-1][j-coin[i]]!=0:
#                 dp[i][j]=dp[i-1][j-coin[i]]+1                
# for i in dp:
#     print(i)
# if dp[-1][-1]!=0:
#     print("min coins: ",dp[-1][-1])
# else:
#     print("No")

# coin change infinite num of coins
# coin=[2,3,4,5]
# amt=12
# dp=[[0]*(amt+1) for i in range(len(coin))]
# for i in range(len(coin)):
#     dp[i][0]=1
# #for first row alone
# for i in range(coin[0],amt+1):
#     dp[0][i]=dp[0][i-coin[0]]
# for i in range(1,len(coin)):
#     # if dp[i][-1]==1:
#     #     print(dp[i][-1])
#     #     break 
#     for j in range(1,amt+1):
#         if j<coin[i]:
#             dp[i][j]=dp[i-1][j]
#         else:
#             dp[i][j]=dp[i-1][j] or dp[i][j-coin[i]]
# for i in dp:
#     print(i)
# if dp[-1][-1]==1:
#     print("True")
# else:
#     print("False")

# min no of coins 1d and infinite
coin=[1,2,5]
k=11
dp=[0]*(k+1)
for i in range(len(coin)):
    # print(dp)
    for j in range(coin[i],k+1):
        if j==coin[i]:
            dp[j]=1
        if dp[j]!=0 and dp[j-coin[i]]!=0:
            dp[j]=min(dp[j],1+dp[j-coin[i]])
        elif dp[j-coin[i]]!=0:
            dp[j]=dp[j-coin[i]]+1
print(dp)
if dp[-1]!=0:
    print("min coins: ",dp[-1])
else:
    print("No")

# given 2 strings print max subsequence length


#Tries
