# # 1. 125. Valid Palindrome o(n)
# def isPalindrome(s):
#     s=s.lower()
#     st=""
#     for i in s:
#         if i.isalnum():
#             st+=i
#     l=0
#     r=len(st)-1
#     while l<r:
#         if st[l]==st[r]:
#             l+=1
#             r-=1        
#         else:
#             return False
#     return True
# s = "A man, a plan, a canal: Panama"
# print(isPalindrome(s))

#     # l=0
#     # r=len(s)-1
#     # while l<r:
#     #     while l<r and not s[l].isalnum():
#     #         l+=1
#     #     while l<r and not s[r].isalnum():
#     #         r-=1
#     #     if s[l].lower()==s[r].lower():
#     #         l+=1
#     #         r-=1
#     #     else:
#     #         return False
#     # return True

# 2. 14. Longest Common Prefix O(n·m·log n + m)
# strs = ["flower","flow","flight"]
# strs.sort()
# res=""
# for i in range(len(strs[0])):
#     if strs[0][i]!=strs[len(strs)-1][i]:
#         break
#     res+=strs[0][i]
# print(res)        

# from os.path import commonprefix
# print(commonprefix(["flower", "flow", "flight"]))

# 3. 242. Valid Anagram 
# O(n)	O(1)
# def isAnagram(s,t):
#     if len(s)!=len(t):
#         return False
#     c={}
#     for ch in s:
#         c[ch]=c.get(ch,0)+1
#     for ch in t:
#         if ch not in c or c[ch]==0:
#             return False
#     return True
# s = "anagram"
# t = "nagaram"
# isAnagram(s,t)

# O(n)	O(n)
    # return Counter(s)==Counter(t)
# O(n log n)	O(n)  
    # s=sorted(s)
    # t=sorted(t)
    # if s==t:
    #     return True
    # return False
# O(n)	O(1)
    # if len(s) != len(t):
    #     return False
    # count = [0] * 26
    # for char in s:
    #     count[ord(char) - ord('a')] += 1
    # for char in t:
    #     if count[ord(char) - ord('a')] == 0:
    #         return False
    #     count[ord(char) - ord('a')] -= 1
    # return True

# 4. 151. Reverse Words in a String
# def reverseWords(s):
#     return " ".join(reversed(s.split()))
# s = "  hello world  "
# print(reverseWords(s))
    # s=s.strip()
    # s=s.split()
    # s.reverse()
    # res=" ".join(map(str,s))
    # return res
   
    # words = s.split()
    # left, right = 0, len(words) - 1
    # while left < right:
    #     words[left], words[right] = words[right], words[left]
    #     left += 1
    #     right -= 1
    # return " ".join(words) 

# 5. 1910. Remove All Occurrences of a Substring
# def removeOccurrences(s, part):
#     while part in s:
#         s = s.replace(part, "", 1) # replace with "" of first occurance of part
#     return s
#     # while True:
#     #     idx = s.find(part)
#     #     if idx == -1:
#     #         break
#     #     s = s[:idx] + s[idx + len(part):]
#     # return s
# s = "daabcbaabcbc"
# part = "abc"
# print(removeOccurrences(s,part))

# 6. 567. Permutation in String
# from collections import Counter
# def checkInclusion(s1,s2):
#     if len(s1) > len(s2):
#         return False
#     s1_count = Counter(s1)
#     window_count = Counter(s2[:len(s1)])
#     if s1_count == window_count:
#         return True
#     for i in range(len(s1), len(s2)):
#         window_count[s2[i]] += 1
#         window_count[s2[i - len(s1)]] -= 1
#         if window_count[s2[i - len(s1)]] == 0:
#             del window_count[s2[i - len(s1)]]
#         if window_count == s1_count:
#             return True
#     return False
# s1 = "ab"
# s2 = "eidbaooo"
# print(checkInclusion(s1,s2))

#    if len(s1) > len(s2):
#        return False
#    s1_count = [0] * 26
#    s2_count = [0] * 26
#    a_ord = ord('a')
#    for i in range(len(s1)):
#        s1_count[ord(s1[i]) - a_ord] += 1
#        s2_count[ord(s2[i]) - a_ord] += 1
#    if s1_count == s2_count:
#        return True
#    for i in range(len(s1), len(s2)):
#        s2_count[ord(s2[i]) - a_ord] += 1
#        s2_count[ord(s2[i - len(s1)]) - a_ord] -= 1
#        if s1_count == s2_count:
#            return True  
#    return False   # f = Counter(s1)

   # ws = len(s1)
   # for i in range(len(s2) - ws + 1):
   #     wf = defaultdict(int)
   #     idx = i
   #     wi = 0
   #     while wi < ws:
   #         wf[s2[idx]] += 1  # ✅ pull from s2
   #         wi += 1
   #         idx += 1
   #     if wf == f:
   #         return True
   # return False

# 7. 443. String Compression
# def compress(chars):
#     n=len(chars)
#     idx=0
#     i=0
#     while i<n:
#         ch=chars[i]
#         count=0
#         while i<n and chars[i]==ch:
#             count+=1
#             i+=1
#         if count==1:
#             chars[idx]=ch
#             idx+=1
#         else:
#             chars[idx]=ch
#             idx+=1
#             for dig in str(count):    # 12->'1','2'
#                 chars[idx]=dig
#                 idx+=1
#     chars[:] = chars[:idx]  #chars.resize(idx)   slice upto idx
#     return idx   # idx pos will be equal to o/p new arr len
# chars = ["a","a","b","b","c","c","c"]
# # chars = ["a"]
# print(chars)
# print(compress(chars))
# print(chars)

# 8. 49. Group Anagrams
# from collections import defaultdict
# def groupAnagrams(strs):
#     dictn=defaultdict(list)
#     for word in strs:
#         wordsort=''.join(sorted(word))
#         dictn[wordsort].append(word)
#     return list(dictn.values())
# strs = ["eat","tea","tan","ate","nat","bat"]
# print(groupAnagrams(strs))

# 76. Minimum Window Substring
# from collections import defaultdict
# def minWindow(s,t):
#     n=len(s)
#     m=len(t)
#     d=defaultdict(int)
#     for i in t:
#         d[i]+=1
#     l=r=c=0
#     si=-1
#     ml=float('inf')
#     required=sum(d.values())
#     while r<n:
#         if d[s[r]]>0:
#             c+=1
#         d[s[r]]-=1
#         while c==required: #c==m:
#             if r-l+1<ml:
#                 ml=r-l+1
#                 si=l
#             d[s[l]]+=1
#             if d[s[l]]>0:
#                 c-=1
#             l+=1
#         r+=1
#     return "" if si==-1 else s[si:si+ml]
# s = "ADOBECODEBANC"
# t = "ABC"
# print(minWindow(s,t))
    # if not t or not s:
    #     return ""
    # need = Counter(t)
    # window = {}
    # have = 0
    # need_count = len(need)
    # l = 0
    # res = [-1, -1]
    # res_len = float("inf")
    # for r in range(len(s)):
    #     ch = s[r]
    #     window[ch] = window.get(ch, 0) + 1
    #     if ch in need and window[ch] == need[ch]:
    #         have += 1
    #     while have == need_count:
    #         # Update result
    #         if (r - l + 1) < res_len:
    #             res = [l, r]
    #             res_len = r - l + 1               
    #         # Shrink from left
    #         window[s[l]] -= 1
    #         if s[l] in need and window[s[l]] < need[s[l]]:
    #             have -= 1
    #         l += 1
    # l, r = res
    # return s[l:r+1] if res_len != float("inf") else ""

# # 10. 1392. Longest Happy Prefix (Knuth-Morris-Pratt (KMP) algorithm) o(n)
# # To solve Longest Happy Prefix, we use the Knuth-Morris-Pratt (KMP) failure table to find the longest prefix that is also a suffix (but not the full string).
# def longestPrefix(s):
#     pre=0
#     suf=1
#     lps=[0]*len(s)
#     while suf<len(s):
#         # Match
#         if s[pre]==s[suf]:
#             lps[suf]=pre+1
#             pre+=1
#             suf+=1
#         # Not Match
#         else:
#             if pre==0:
#                 lps[suf]=0
#                 suf+=1
#             else:
#                 pre=lps[pre-1]
#     # return lps[-1]
#     return s[:lps[-1]]
# s="ababab"
# print(longestPrefix(s))

# # 11. 28. Find the Index of the First Occurrence in a String
# class Solution:
#     def findlps(self,s,lps):
#         pre=0
#         suf=1
#         while suf<len(s):
#             # Match
#             if s[pre]==s[suf]:
#                 lps[suf]=pre+1
#                 pre+=1
#                 suf+=1
#             # Not Match
#             else:
#                 if pre==0:
#                     lps[suf]=0
#                     suf+=1
#                 else:
#                     pre=lps[pre-1]
#     def strStr(self, haystack: str, needle: str) -> int:
#         #kmp
#         if not needle:
#             return 0 
#         lps=[0]*len(needle)
#         self.findlps(needle,lps)
#         first=second=0
#         while first<len(haystack) and second<len(needle):
#             # match
#             if haystack[first]==needle[second]:
#                 first+=1
#                 second+=1
#             # not match
#             else:
#                 if second==0:
#                     first+=1
#                 else:
#                     second=lps[second-1]
#         if second==len(needle):
#             return first-second
#         return -1
    
#         # if not needle:
#         #     return 0
#         # for i in range(len(haystack) - len(needle) +1):
#         #     if haystack[i:i+len(needle)]==needle:
#         #         return i
#         # return -1
# l=Solution()
# haystack = "sadbutsad"
# needle = "sad"
# print(l.strStr(haystack,needle))

# KMP Algorithm (Knuth-Morris-Pratt) – Pattern Matching with Zero Backtracking
# KMP finds the first occurrence of a pattern P in a text T without rechecking characters using a clever precomputed array — the LPS (Longest Prefix Suffix) array.

# Robin Karp algorithm (Fast Substring Search using Hashing)
# Efficiently finds the first (or all) occurrence(s) of a pattern P in a text T using rolling hash.
# Instead of comparing substrings character-by-character (like brute-force), it compares hash values, drastically reducing unnecessary checks.