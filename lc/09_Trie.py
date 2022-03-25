# -- coding: utf-8 --

class Trie:
    """
    前缀树
    """
    def __init__(self):
        self.children = [None] * 26
        self.is_end = False

    def insert(self, word: str):
        if not word:
            return
        # 指针指向当前节点，从根节点开始插入字符
        p_node = self
        for ch in word:
            idx = ord(ch) - ord('a')
            if not p_node.children[idx]:
                p_node.children[idx] = Trie()
            # 指针指向当前节点的子节点
            p_node = p_node.children[idx]
        p_node.is_end = True

    def search_prefix(self, prefix: str):
        if not prefix:
            return None
        # 指针指向当前节点，从根节点开始遍历
        p_node = self
        for ch in prefix:
            idx = ord(ch) - ord('a')
            if not p_node.children[idx]:
                return None
            # 指针指向当前节点的子节点
            p_node = p_node.children[idx]
        return p_node

    def search(self, word: str):
        p_node = self.search_prefix(word)
        return p_node is not None and p_node.is_end

    def startsWith(self, prefix: str):
        p_node = self.search_prefix(prefix)
        return p_node is not None


if __name__ == '__main__':
    trie = Trie()
    print(trie.search("a"))
