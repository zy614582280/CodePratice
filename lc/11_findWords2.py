# -- coding: utf-8 --
from typing import List


def findWords(board, words):
    """
    单词搜索 II
    :type board: List[List[str]]
    :type words: List[str]
    :rtype: List[str]
    """
    result = []
    rows = len(board)
    cols = len(board[0])
    for word in words:
        visited = [False] * cols * rows
        flag = False
        for r in range(rows):
            if flag:
                break
            for c in range(cols):
                if flag:
                    break
                if find_word_score(board, word, visited, 0, c, r, cols, rows):
                    result.append(word)
                    flag = True
    return result


def find_word_score(matrix: List[List[str]],
                    word: str,
                    visited: List[bool],
                    char_idx: int,
                    c: int,
                    r: int,
                    cols: int,
                    rows: int):
    if char_idx == len(word):
        return True
    flag = False
    if 0 <= c < cols and 0 <= r < rows \
            and matrix[r][c] == word[char_idx] and not visited[cols * r + c]:
        char_idx += 1
        visited[cols * r + c] = True
        # 递归判断 (r, c-1)、(r, c+1)、(r-1,c)、(r+1, c)
        flag = find_word_score(matrix, word, visited, char_idx, c - 1, r, cols, rows) \
               or find_word_score(matrix, word, visited, char_idx, c + 1, r, cols, rows) \
               or find_word_score(matrix, word, visited, char_idx, c, r - 1, cols, rows) \
               or find_word_score(matrix, word, visited, char_idx, c, r + 1, cols, rows)
        if not flag:
            char_idx -= 1
            visited[cols * r + c] = False
    return flag


if __name__ == '__main__':
    print(findWords(board=[["o", "a", "a", "n"], ["e", "t", "a", "e"], ["i", "h", "k", "r"], ["i", "f", "l", "v"]],
                    words=["oath", "pea", "eat", "rain"]))

    print(findWords(board=[["a", "b"], ["c", "d"]],
                    words=["abcb"]))
