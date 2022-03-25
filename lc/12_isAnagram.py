# -- coding: utf-8 --

def isAnagram(s, t):
    """
    有效的字母异位词
    :type s: str
    :type t: str
    :rtype: bool
    """
    if len(s) != len(t):
        return False

    char_count = {}
    for char in s:
        char_count[char] = char_count.get(char, 0) + 1
    for char in t:
        char_count[char] = char_count.get(char, 0) - 1

    for k, v in char_count.items():
        if v != 0:
            return False
    return True


if __name__ == '__main__':
    print(isAnagram(s="anagram", t="nagaram"))
    print(isAnagram(s="rat", t="car"))
