# -- coding: utf-8 --

def wordBreak(s, wordDict):
    """
    :type s: str
    :type wordDict: List[str]
    :rtype: bool
    单词拆分
    动态规划
    """
    dp = [False] * (len(s) + 1)  # dp[i] 表示 s[:i]能否拆分
    dp[0] = True  # dp[0] 表示 ''
    for i in range(0, len(s) + 1):
        # 从前往后依次遍历字符串
        for j in range(i, -1, -1):
            # 从i开始，从后往前遍历字符串s[0:i]
            # 前半段s[:j]是否在wordDict
            if not dp[j]:
                continue
            # 后半段s[j:i]是否在wordDict
            suffix = s[j:i]
            if suffix in wordDict:
                # 后半段在wordDict，dp[i]设为True，不用在遍历后半段了
                dp[i] = True
                break
    return dp[len(s)]


if __name__ == '__main__':
    print(wordBreak(s="leetcode", wordDict=["leet", "code"]))
    print(wordBreak(s="applepenapple", wordDict=["apple", "pen"]))
    print(wordBreak(s="catsandog", wordDict=["cats", "dog", "sand", "and", "cat"]))
