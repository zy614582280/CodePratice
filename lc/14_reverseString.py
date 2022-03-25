# -- coding: utf-8 --

def reverseString(s):
    """
    反转字符串
    :type s: List[str]
    :rtype: None Do not return anything, modify s in-place instead.
    """
    if not s or len(s) == 1:
        return

    p1, p2 = 0, len(s) - 1
    while p1 < p2:
        tmp = s[p2]
        s[p2] = s[p1]
        s[p1] = tmp
        p1 += 1
        p2 -= 1


if __name__ == '__main__':
    s = ["h", "e", "l", "l", "o"]
    reverseString(s)
    print(s)

    s = ["H", "a", "n", "n", "a", "h"]
    reverseString(s)
    print(s)

