#    tries
# insertion
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.count = 0  # How often this node is visited

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.count += 1  # Count usage
        node.is_end = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end

    def autocomplete(self, prefix):
        def collect_words(node, current_word):
            words = []
            if node.is_end:
                words.append((current_word, node.count))
            for char, child in node.children.items():
                words += collect_words(child, current_word + char)
            return words

        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        suggestions = collect_words(node, prefix)
        suggestions.sort(key=lambda x: -x[1])  # Sort by usage frequency
        return [word for word, _ in suggestions]

# Example
trie = Trie()
for word in ["hello", "helium", "hero", "heron", "hex", "hello", "hero"]:
    trie.insert(word)

print(trie.autocomplete("he"))  # ['hello', 'hero', 'helium', 'heron', 'hex']
print(trie.search("hero"))      # True
print(trie.search("her"))       # False
