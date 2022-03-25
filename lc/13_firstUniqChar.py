# -- coding: utf-8 --

def firstUniqChar(s):
    """
    字符串中的第一个唯一字符
    :type s: str
    :rtype: int
    """
    # 计数的问题
    if not s:
        return -1
    char_counts = {}
    for ch in s:
        char_counts[ch] = char_counts.get(ch, 0) + 1
    for i in range(len(s)):
        if char_counts[s[i]] == 1:
            return i
    return -1


if __name__ == '__main__':
    print(firstUniqChar(s="leetcode"))
    print(firstUniqChar(s="loveleetcode"))
    print(firstUniqChar(s="aabb"))
