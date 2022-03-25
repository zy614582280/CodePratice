# -- coding: utf-8 --
import re


def isPalindrome(s):
    """
    判断字符串回文
    :type s: str
    :rtype: bool
    """
    if not s:
        return False
    p1, p2 = 0, len(s) - 1
    while p1 < p2:
        char_p1 = s[p1].lower()
        char_p2 = s[p2].lower()
        if not re.match('[a-zA-Z0-9]', char_p1):
            p1 += 1
            continue
        if not re.match('[a-zA-Z0-9]', char_p2):
            p2 -= 1
            continue
        if char_p1 == char_p2:
            p1 += 1
            p2 -= 1
        else:
            return False
    return True


if __name__ == '__main__':
    print(isPalindrome("A man, a plan, a canal: Panama"))
    print(isPalindrome("race a car"))

