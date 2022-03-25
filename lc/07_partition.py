# -- coding: utf-8 --

from typing import List


def partition(s):
    """
    分割回文串
    :type s: str
    :rtype: List[List[str]]
    """

result = []


def dfs(s: str,
        remain: List,
        left: int):
    print()
    if left == len(s):
        result.extend(remain)
        print(f"extend {remain}")
        return
    for right in range(left, len(s)):
        print(f"left={left}, right={right}, remain={remain}")
        if isPalindrome(s, left, right):
            remain.append(s[left: right + 1])
            print(f"left={left}, right={right}, remain={remain}")
            dfs(s, remain, right + 1)
            item = remain.pop()
            print(f"pop {item}")
            print(f"left={left}, right={right}, remain={remain}")

def isPalindrome(s: str,
                 left: int,
                 right: int):
    if left >= right:
        return True
    return s[left] == s[right] and isPalindrome(s, left + 1, right - 1)


if __name__ == '__main__':
    dfs("aac", [], 0)
    print(result)
