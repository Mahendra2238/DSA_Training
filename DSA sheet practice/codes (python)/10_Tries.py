# 14. Longest Common Prefix
# Time Complexity: O(N*M) where N is number of strings and M is length of smallest string / O(Σ|s|)
# Space Complexity: O(N*M) for trie storage / O(Σ|s|) for trie storage
from typing import List
class Tnode:
    def __init__(self):
        self.data={}
        self.next=False
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        def insert(node,word):
            for ch in word:
                if ch not in node.data:
                    node.data[ch]=Tnode()
                node=node.data[ch] 
            node.next=True
        def lcp(node):
            res=[]
            while(1):
                if len(node.data)!=1 or node.next:
                    return ''.join(res)
                for i in node.data:
                    res.append(i)
                    node=node.data[i]
        node=Tnode()
        for s in strs:
            insert(node,s)
        return lcp(node)

# class Solution:
#     def longestCommonPrefix(self, strs: List[str]) -> str:
#         strs.sort()
#         res=""
#         for i in range(len(strs[0])):
#             if strs[0][i]!=strs[len(strs)-1][i]:
#                 break
#             res+=strs[0][i]
#         return res        

# 15. Word Break
# Time Complexity: O(N^3) where N is length of string s
# Space Complexity: O(N) for memoization
from typing import List
class Tnode:
    def __init__(self):
        self.data={}
        self.next=False
class Solution:
    def wordBreak(self,s:str,wordDict:List[str])->bool:
        root=Tnode();memo={}
        def insert(node,word):
            for ch in word:
                if ch not in node.data:
                    node.data[ch]=Tnode()
                node=node.data[ch]
            node.next=True
        def search(node,key):
            for ch in key:
                if ch not in node.data:
                    return False
                node=node.data[ch]
            return node.next
        for w in wordDict:
            insert(root,w)
        def dfs(s):
            if s=="":
                return True
            if s in memo:
                return memo[s]
            for i in range(1,len(s)+1):
                if search(root,s[:i]) and dfs(s[i:]):
                    memo[s]=True
                    return True
            memo[s]=False
            return False
        return dfs(s)

# Phone directory gfg
class Tnode:
    def __init__(self):
        self.data={}      # children nodes
        self.end=False    # end of contact

class Solution:
    def displayContacts(self, n, contact, s):
        root=Tnode()
        def insert(word):
            node=root
            for ch in word:
                if ch not in node.data:
                    node.data[ch]=Tnode()   # create trie node
                node=node.data[ch]
            node.end=True                  # mark end of word
        for c in contact:
            insert(c)                      # build trie
        def collect(node,prefix,res):
            if node.end:
                res.append(prefix)         # valid contact
            for ch in node.data:
                collect(node.data[ch],prefix+ch,res)
        ans=[];prefix="";node=root
        for ch in s:
            prefix+=ch
            if node and ch in node.data:
                node=node.data[ch]         # move in trie
                res=[]
                collect(node,prefix,res)   # collect matches
                ans.append(sorted(res))    # store result
            else:
                node=None
                ans.append(["0"])          # no match
        return ans
n = 3
contact= ["geeikistest", "geeksforgeeks", "geeksfortest"]
s = "geeips"
obj = Solution()
print(obj.displayContacts(n, contact, s))

# Time Complexity:
# Trie build: O(total characters in contact list)
# Search + collect: O(p * k) where p = len(s), k = matched contacts

# Space Complexity:
# Trie storage: O(total characters in contact list)
# Recursion stack (collect): O(max contact length)

# 16. Implement Trie (Prefix Tree)
# Time Complexity: O(M) for insert, search, startsWith where M is length of
# Space Complexity: O(Σ|s|) for trie storage where Σ|s| is sum of lengths of all words inserted

class Tnode:
    def __init__(self):
        self.data={}      # children
        self.end=False    # end of word

class Trie:
    def __init__(self):
        self.root=Tnode() # initialize root

    def insert(self, word: str) -> None:
        node=self.root
        for ch in word:
            if ch not in node.data:
                node.data[ch]=Tnode()   # create node
            node=node.data[ch]
        node.end=True                  # mark word end

    def search(self, word: str) -> bool:
        node=self.root
        for ch in word:
            if ch not in node.data:
                return False           # path missing
            node=node.data[ch]
        return node.end                # must end here

    def startsWith(self, prefix: str) -> bool:
        node=self.root
        for ch in prefix:
            if ch not in node.data:
                return False           # prefix missing
            node=node.data[ch]
        return True
    
# Trie Node
class Tnode:
    def __init__(self):
        self.data={}     # children
        self.end=False   # end of word

class Trie:
    def __init__(self):
        self.root=Tnode()

    def insert(self,word):
        node=self.root
        for ch in word:
            if ch not in node.data:
                node.data[ch]=Tnode()
            node=node.data[ch]
        node.end=True

    def allPrefixesExist(self,word):
        node=self.root
        for ch in word:
            node=node.data.get(ch)
            if not node or not node.end:   # prefix missing
                return False
        return True

class Solution:
    def longestValidWord(self,words):
        trie=Trie()
        for w in words:
            trie.insert(w)                 # build trie
        ans=""
        for w in words:
            if trie.allPrefixesExist(w):   # check prefixes
                if len(w)>len(ans) or (len(w)==len(ans) and w<ans):
                    ans=w                  # update answer
        return ans
        
# class Solution:
#     def longestValidWord(self, words):
#         # build trie
#         trie = Trie()
#         for w in words:
#             trie.insert(w)

#         ans = ""
#         for w in words:
#             valid = True
#             pref = ""
#             for ch in w:
#                 pref += ch
#                 if not trie.search(pref):   # every prefix must exist
#                     valid = False
#                     break
#             if valid:
#                 if len(w) > len(ans) or (len(w) == len(ans) and w < ans):
#                     ans = w
#         return ans

