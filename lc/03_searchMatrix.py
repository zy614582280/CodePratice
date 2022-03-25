# -- coding: utf-8 --
"""
搜索二维矩阵 II

编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：

每行的元素从左到右升序排列。
每列的元素从上到下升序排列。

输入：matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 5
输出：true

输入：matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 20
输出：false

"""


def searchMatrix(matrix, target):
    """
    :type matrix: List[List[int]]
    :type target: int
    :rtype: bool
    从右上角往左下角找
    """
    if not matrix:
        return False
    row = len(matrix)
    col = len(matrix[0])
    r, c = 0, col - 1
    while r < row and c >= 0:
        num = matrix[r][c]
        if num == target:
            return True
        elif num > target:
            c -= 1
        else:
            r += 1
    return False


def searchMatrix2(matrix, target):
    """
    :type matrix: List[List[int]]
    :type target: int
    :rtype: bool
    从左下角往右上角找
    """
    if not matrix:
        return False
    row, col = len(matrix), len(matrix[0])
    r, c = row - 1, 0
    while r >= 0 and c <= col - 1:
        num = matrix[r][c]
        if num == target:
            return True
        elif num > target:
            r -= 1
        else:
            c += 1
    return False


if __name__ == '__main__':
    print(searchMatrix(
        matrix=[[1, 4, 7, 11, 15], [2, 5, 8, 12, 19], [3, 6, 9, 16, 22], [10, 13, 14, 17, 24], [18, 21, 23, 26, 30]],
        target=5))

    print(searchMatrix(
        matrix=[[1, 4, 7, 11, 15], [2, 5, 8, 12, 19], [3, 6, 9, 16, 22], [10, 13, 14, 17, 24], [18, 21, 23, 26, 30]],
        target=20))

    print(searchMatrix2(
        matrix=[[1, 4, 7, 11, 15], [2, 5, 8, 12, 19], [3, 6, 9, 16, 22], [10, 13, 14, 17, 24], [18, 21, 23, 26, 30]],
        target=5))

    print(searchMatrix2(
        matrix=[[1, 4, 7, 11, 15], [2, 5, 8, 12, 19], [3, 6, 9, 16, 22], [10, 13, 14, 17, 24], [18, 21, 23, 26, 30]],
        target=20))
