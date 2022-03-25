# -- coding: utf-8 --
"""
无重复字符的最长子串
给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。
"""


def lengthOfLongestSubstring(s: str):
    """
    :type s: str
    :rtype: int
    双指针，滑窗，滑窗用队列实现
    """
    if not s:
        return 0
    max_len = 0
    queue = []
    for i in range(len(s)):
        while s[i] in queue:
            queue.pop(0)
        queue.append(s[i])
        max_len = max(max_len, len(queue))
    return max_len


if __name__ == '__main__':
    print(lengthOfLongestSubstring('abcabcbb'))
    print(lengthOfLongestSubstring('bbbbb'))
    print(lengthOfLongestSubstring('pwwkew'))