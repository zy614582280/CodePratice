# -- coding: utf-8 --
"""
题目：找到数组中任一重复的数字
在一个长度为n的数组中，所有的数字都是0~n-1的范围内。
数组中某些数字是重复的，但不知道有几个数字是重复的，也不知道每个数字都重复了几次。
请找出任一重复的数字。

eg. [2, 3, 1, 0, 2, 5, 3]
return 2 or 3.

思路：
数组长度为n，每个数都在0~n-1范围内，如果每个数都不重复，那么数组排序之后，每个数字与其下标相等，[0, 1, 2, 3]
我们逆向思维，对数组进行从前向后排序，对于索引i，比较当前值l[i]是否与其相等，如果相等，则继续对比下一个索引；
如果不相等，则比较l[l[i]]跟l[i]是否相等，如果不相等，则交换l[l[i]]跟l[i]的位置（这一步操作使得l[l[i]] = l[i]）
如果相等，恭喜你，已经找到了答案。

"""
from typing import List


def find_repeat_num(fuc_input):
    if not fuc_input:
        return -1
    i = 0
    while i < len(fuc_input):
        print(fuc_input)
        if i == fuc_input[i]:
            i += 1
        else:
            if fuc_input[i] == fuc_input[fuc_input[i]]:
                return fuc_input[i]
            else:
                # exchange fuc_input[i] fuc_input[fuc_input[i]]
                tmp = fuc_input[i]
                fuc_input[i] = fuc_input[fuc_input[i]]
                fuc_input[tmp] = tmp
                i += 1
    return -1


"""
折半查找
"""


def bisearch(input_list, value):
    if not input_list or not value:
        return -1
    # 保证列表是有序的
    input_list.sort()
    start = 0
    end = len(input_list) - 1
    while start <= end:
        mid = (start + end) // 2
        # 查中间
        if input_list[mid] == value:
            return mid
        if input_list[mid] > value:
            # 查左边
            end = mid - 1
        else:
            # 查右边
            start = mid + 1
    return -1


"""
不修改列表找出重复的数字

在一个长度为n的数组里，所有的数都是1~n-1之间，至少有一个数字重复，找出任意一个重复的数组，但不能修改原来的数组。
例如： [2, 3, 5, 4, 3, 2, 6, 7] 
返回： 2 或者 3

思路：
如果1~n 之间没有重复的数，那么1~n之间的数一共有n个，如果超过n个，那么一定存在重复的数
start = 1 (最小数)
end = len (最大数）
mid = (start + end) // 2 （中间数）

类似于折半查找，我们每次先查中间数mid的计数，如果count(mid) > 1，则mid重复；
否则数start~mid-1之间的计数，如果count(start~ mid-1) > mid-start，则start~mid-1之间存在重复；
否则mid+1~end之间存在重复
"""


def find_repeat_num2(input_list):
    if not input_list:
        return -1
    start = 1
    end = len(input_list)
    while start <= end:
        mid = (start + end) // 2
        count_ = count(input_list, mid, mid)
        if count_ > 1:
            return mid
        count_ = count(input_list, start, mid - 1)
        if count_ > mid - start:
            end = mid - 1
        else:
            start = mid + 1
    return -1


def count(input_list, start, end):
    count_ = 0
    if end < start or not input_list:
        return count_
    for i in input_list:
        if start <= i <= end:
            count_ += 1
    return count_


"""
二维有序数组里查找

在一个二维数组里，每一行从左往右递增，每一列从从上到下递增。
输入一个二维数组，判断是否存在某个数。
例如：
[[1, 2, 8, 9],
 [2, 4, 9, 12],
 [4, 7, 10, 13],
 [6, 8, 11, 15]]
输入7，存在返回true；输入5，不存在返回false
"""


def find_matrix_value(matrix, value):
    if not matrix or not value:
        return False
    rows = len(matrix)
    columns = len(matrix[0])

    row = 0
    col = columns - 1
    while row <= rows - 1 and col >= 0:
        if matrix[row][col] == value:
            return True
        if matrix[row][col] > value:
            col = col - 1
        else:
            row = row + 1
    return False


def find_matrix_value2(matrix, value):
    if not matrix or not value:
        return False
    rows = len(matrix)
    columns = len(matrix[0])

    row = rows - 1
    col = 0
    while row >= 0 and col <= columns - 1:
        if matrix[row][col] == value:
            return True
        if matrix[row][col] > value:
            row = row - 1
        else:
            col = col + 1
    return False


def replace_blank(input_str):
    """
    把输入字符串中的空格都替换成"%20"，要求时间复杂度为O（n)
    思路：从后往前替换，先计算出替换后的字符串长度，然后逐个将字符移动到替换后的位置
    :param input_str: We are happy.
    :return: We%20are%20happy.
    """
    if not input_str:
        return input_str
    blank_count = 0
    for s in input_str:
        if s == " ":
            blank_count += 1
    if blank_count == 0:
        return input_str
    # python中字符串是不可变类型，不能在原先字符串上直接修改字符
    # 这里把字符串改为字符列表
    input_str = list(input_str)

    p1 = len(input_str) - 1
    p2 = p1 + blank_count * 2
    input_str = input_str + [" "] * blank_count * 2
    while p2 > p1 >= 0:
        if input_str[p1] == " ":
            input_str[p2] = "0"
            p2 -= 1
            input_str[p2] = "2"
            p2 -= 1
            input_str[p2] = "%"
            p2 -= 1
        else:
            input_str[p2] = input_str[p1]
            p2 -= 1
        p1 -= 1
    return "".join(input_str)


def merge_sorted_list(a1, a2):
    """
    合并两个有序的数组a1，a2，
    思路：从后往前合并
    :param a1:
    :param a2:
    :return:
    """
    if not a1:
        return a2
    if not a2:
        return a1

    result = [0] * (len(a1) + len(a2))
    p1 = len(a1) - 1
    p2 = len(a2) - 1
    p = len(result) - 1
    # 从后往前，比较a1、a2的后半部分
    while p1 >= 0 and p2 >= 0:
        if a1[p1] >= a2[p2]:
            result[p] = a1[p1]
            p1 -= 1
        else:
            result[p] = a2[p2]
            p2 -= 1
        p -= 1
    # 取剩余的部分
    while p1 >= 0:
        result[p] = a1[p1]
        p -= 1
        p1 -= 1
    while p2 >= 0:
        result[p] = a2[p2]
        p -= 1
        p2 -= 1

    return result


class BinaryTreeNode:
    def __init__(self, root, lchild, rchild):
        self.root = root
        self.lchild = lchild
        self.rchild = rchild


def preorder(tree: BinaryTreeNode,
             preorderlist=[]):
    if not tree or not tree.root:
        return
    preorderlist.append(tree.root)
    if tree.lchild:
        preorder(tree.lchild, preorderlist)
    if tree.rchild:
        preorder(tree.rchild, preorderlist)


def midorder(tree: BinaryTreeNode,
             midorderlist=[]):
    if not tree or not tree.root:
        return
    if tree.lchild:
        midorder(tree.lchild, midorderlist)
    midorderlist.append(tree.root)
    if tree.rchild:
        midorder(tree.rchild, midorderlist)


def bfsorder(tree: BinaryTreeNode,
             bfsorderlist=[]):
    """
    树的宽度优先遍历
    :param tree:
    :param bfsorderlist:
    :return:
    """
    queues = []
    bfsorderlist.append(tree.root)
    if tree.lchild:
        queues.append(tree.lchild)
    if tree.rchild:
        queues.append(tree.rchild)

    while len(queues) > 0:
        for node in queues:
            bfsorderlist.append(node.root)
            if node.lchild:
                queues.append(node.lchild)
            if node.rchild:
                queues.append(node.rchild)


def build_binary_tree(preorderlist,
                      midorderlist):
    if not preorderlist or not midorderlist:
        return None

    root_index = midorderlist.index(preorderlist[0])
    left_len = root_index
    lchild = None
    if left_len > 0:
        # 左子树
        lchild = build_binary_tree(preorderlist[1: root_index + 1],
                                   midorderlist[0: root_index])

    rchild = None
    right_len = len(preorderlist) - root_index - 1
    if right_len > 0:
        # 右子树
        rchild = build_binary_tree(preorderlist[root_index + 1:],
                                   midorderlist[root_index + 1:])

    return BinaryTreeNode(preorderlist[0],
                          lchild,
                          rchild)


class BinaryTreeNode2(BinaryTreeNode):
    def __init__(self, root, lchild, rchild, parent):
        super(BinaryTreeNode2, self).__init__(root, lchild, rchild)
        self.parent = parent


def find_next_point(node: BinaryTreeNode2):
    if node is None:
        return None
    # 如果节点有右子树，则下一个节点是右子树的最左节点
    if node.rchild is not None:
        next_node = node.rchild
        while next_node.lchild is not None:
            next_node = next_node.lchild
        return next_node
    # 如果节点没有右子树，则从该节点向上遍历，如果某个节点是其父节点的左子节点，则其父节点是下一个节点
    current_node = node
    parent_node = node.parent
    next_node = None
    while parent_node is not None and current_node == parent_node.lchild:
        current_node = parent_node
        parent_node = current_node.patent
    return next_node


class QueueWithTwoStacks:
    """
    两个栈实现队列功能，先进先出
    """

    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def is_empty(self):
        if self.stack1 or self.stack2:
            return False
        return True

    def __len__(self):
        return len(self.stack1) + len(self.stack2)

    def append_tail(self, value):
        self.stack1.append(value)

    def delete_head(self):
        if len(self):
            return None

        if not self.stack2:
            while self.stack1:
                value = self.stack1.pop()
                self.stack2.append(value)
        return self.stack2.pop()


class Queue:
    def __init__(self):
        self.__queue = []

    def __len__(self):
        return len(self.__queue)

    def is_empty(self):
        return len(self) == 0

    def enqueue(self, value):
        self.__queue.append(value)

    def dequeue(self):
        if self.__queue:
            return self.__queue.pop(0)
        return None


class StackWithTwoQueue:
    def __init__(self):
        self.queue1 = Queue()
        self.queue2 = Queue()

    def __len__(self):
        return len(self.queue1) + len(self.queue2)

    def is_empty(self):
        return len(self) == 0

    def push(self, value):
        self.queue1.enqueue(value)

    def pop(self):
        if self.is_empty():
            return None

        if len(self.queue1) > 0:
            while len(self.queue1) > 1:
                value = self.queue1.dequeue()
                self.queue2.enqueue(value)
            return self.queue1.dequeue()
        else:
            while len(self.queue2) > 1:
                value = self.queue2.dequeue()
                self.queue1.enqueue(value)
            return self.queue2.dequeue()

    def push(self, value):
        self.queue1.enqueue(value)


def fibonacci(n):
    """
    斐波那契数列
    :param n:
    :return:
    """
    if n == 0:
        return 0
    if n == 1:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)


def fibonacci2(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    last_one = 1
    last_two = 0
    for i in range(2, n + 1):
        tmp = last_two + last_one
        last_two = last_one
        last_one = tmp
    return last_one


def swap(a, b):
    tmp = a
    a = b
    b = tmp


def quick_sort(arr, start, end):
    if start > end:
        return
    pivot = arr[start]
    i = start
    j = end
    while i != j:
        while arr[j] >= pivot and j > i:
            j = j - 1
        while arr[i] <= pivot and j > i:
            i = i + 1
        # 交换 start、end 元素
        arr[i], arr[j] = arr[j], arr[i]

    # 交换 start、pivot元素
    arr[start], arr[i] = arr[i], arr[start]
    # 递归
    quick_sort(arr, start, i - 1)
    quick_sort(arr, i + 1, end)


def quick_sort2(arr, begin, end):
    if end < begin:
        return
    pivot = arr[begin]
    i = begin
    j = end
    while i != j:
        while arr[j] >= pivot and j > i:
            j -= 1

        while arr[i] <= pivot and j > i:
            i += 1
        # 交换i、j
        arr[i], arr[j] = arr[j], arr[i]
    # 交换 begin、 i
    arr[i], arr[begin] = arr[begin], arr[i]
    # 递归
    quick_sort2(arr, begin, i - 1)
    quick_sort2(arr, i + 1, end)


def min_revolved(arr):
    """
    查找旋转数组的最小值
    比如： [3, 4, 5, 6, 7, 8, 0, 1, 2] -> 0
    二分查找
    :param arr:
    :return:
    """
    if not arr:
        return None
    i = 0
    j = len(arr) - 1
    if arr[j] > arr[i]:
        return arr[i]
    while arr[i] >= arr[j]:
        if i + 1 == j:
            return arr[j]
        mid = (i + j) // 2
        if arr[mid] == arr[i] == arr[j]:
            # 老老实实顺序查询
            min_value = arr[0]
            for m in range(i, j + 1):
                if arr[m] < min_value:
                    min_value = arr[m]
            return min_value
        elif arr[mid] >= arr[i]:
            i = mid
        elif arr[mid] <= arr[j]:
            j = mid


def has_path(matrix, path):
    """
    矩阵中的路径
    例如, [[a, b, c, d],
          [e, f, g, h],
          [h, i ,j ,k]]
    输入bfgj,返回true
    输入bfgi，返回false
    :param matrix:
    :return:
    """
    if not matrix:
        return False
    rows = len(matrix)
    cols = len(matrix[0])
    visited = [False] * rows * cols
    path_index = 0
    for r in range(rows):
        for c in range(cols):
            if has_path_core(matrix, r, c, rows, cols, path, path_index, visited):
                return True
    return False


def has_path_core(matrix, r, c, rows, cols, path, path_index, visited):
    if path_index == len(path):
        return True
    flag = False
    try:
        if 0 <= r < rows and 0 <= c < cols and path[path_index] == matrix[r][c] and not visited[r * cols + c]:
            path_index += 1
            visited[r * cols + c] = True
            # 递归判断 (r, c-1)、(r, c+1)、(r-1,c)、(r+1, c)
            flag = has_path_core(matrix, r, c - 1, rows, cols, path, path_index, visited) \
                   or has_path_core(matrix, r, c + 1, rows, cols, path, path_index, visited) \
                   or has_path_core(matrix, r - 1, c, rows, cols, path, path_index, visited) \
                   or has_path_core(matrix, r + 1, c, rows, cols, path, path_index, visited)
            if not flag:
                path_index -= 1
                visited[r * cols + c] = False
    except:
        print(r, c, path_index, r * cols + c)
    return flag


def robot_moving_count(threshold: int,
                       rows: int,
                       cols: int):
    """
    机器人的运动范围
    :param threshold:
    :param rows:
    :param cols:
    :return:
    """
    visited = [False] * rows * cols
    count = robot_moving_count_core(0, 0, rows, cols, threshold, visited)
    return count


def robot_moving_count_core(r,
                            c,
                            rows,
                            cols,
                            threshold,
                            visited,
                            ):
    count = 0
    if check_row_col(r, c, rows, cols, threshold, visited):
        print(f"r={r}, c={c}, count={count}")
        visited[r * cols + c] = True
        count += 1
        count += robot_moving_count_core(r, c+1, rows, cols, threshold, visited)  # 上
        count += robot_moving_count_core(r, c-1, rows, cols, threshold, visited)  # 下
        count += robot_moving_count_core(r-1, c, rows, cols, threshold, visited)  # 左
        count += robot_moving_count_core(r+1, c, rows, cols, threshold, visited)  # 右
    return count


def check_row_col(r,
                  c,
                  rows,
                  cols,
                  threshold,
                  visited):
    if 0 <= r < rows \
            and 0 <= c < cols \
            and sum_num(r) + sum_num(c) <= threshold \
            and not visited[r * cols + c]:
        return True
    return False


def sum_num(n):
    result = 0
    while n >= 1:
        result += n % 10
        n = n // 10
    return result


def max_product_after_cut(length: int):
    """
    剪绳子
    动态规划法
    长度为n的绳子，切断后的最大乘积设为f(n)
    那么则有 f(n) = max(i*(n-i), i*f(n-i))
    :param length:
    :return:
    """
    if length < 2:
        return 0
    if length == 2:
        return 1
    if length == 3:
        return 2

    products = [0] * (length + 1)
    products[0] = 0
    products[1] = 0
    products[2] = 1
    products[3] = 2

    for i in range(4, length + 1):
        max_product = 0
        for j in range(1, i // 2 + 1):
            max_product = max(max_product, j * products[i - j], j * (i - j))
            products[i] = max_product
    return products[length]


def number_of1(n: int):
    """
    输入一个整数，输出其二进制数的1的个数
    比如，8 ，二进制为1000，有一个1
    :param n:
    :return:
    """
    if n < 0:
        n = n & 0xFFFFFFFF
    num_1 = 0
    while n > 0:
        n &= n-1
        num_1 += 1
    return num_1


if __name__ == '__main__':
    # print(find_repeat_num([2, 3, 1, 0, 2, 5, 3]))
    # print(find_repeat_num([0, 1, 2, 3, 4, 5]))
    # print(find_repeat_num([0, 0, 0, 0, 0, 0]))

    # print(bisearch([0, 1, 2, 3, 4, 5], 8))

    # print(find_repeat_num2([2, 3, 5, 4, 3, 2, 6, 7]))
    # print(find_repeat_num2([0, 1, 2, 3, 4, 5]))
    # print(find_repeat_num2([1, 1, 1, 1, 1, 1]))

    # print(find_matrix_value2([[1, 2, 8, 9],
    #                          [2, 4, 9, 12],
    #                          [4, 7, 10, 13],
    #                          [6, 8, 11, 15]], 7))
    # print(find_matrix_value2([[1, 2, 8, 9],
    #                          [2, 4, 9, 12],
    #                          [4, 7, 10, 13],
    #                          [6, 8, 11, 15]], 5))

    # print(replace_blank("We are   happy."))
    # print(merge_sorted_list([1, 2, 3, 4],
    #                         [3, 4, 5, 6]))

    # tree = build_binary_tree([1, 2, 4, 7, 3, 5, 6, 8],
    #                          [4, 7, 2, 1, 5, 3, 8, 6])
    # preorderlist = []
    # preorder(tree, preorderlist)
    # print(preorderlist)
    #
    # midorderlist = []
    # midorder(tree, midorderlist)
    # print(midorderlist)

    # queue = QueueWithTwoStacks()
    # queue.append_tail(0)
    # queue.append_tail(1)
    # queue.append_tail(2)
    # while len(queue) > 0:
    #     print(queue.delete_head())

    # stack = StackWithTwoQueue()
    # for i in [1, 2, 3, 4, 5]:
    #     stack.push(i)
    # while len(stack) > 0:
    #     print(stack.pop())

    # import time
    # start_time = time.time()
    # fn = fibonacci(3)
    # print(f"cost time: {time.time() - start_time}, value: {fn}")
    # start_time = time.time()
    # fn2 = fibonacci2(3)
    # print(f"cost time: {time.time() - start_time}, value: {fn2}")

    # arr = [6, 1, 2, 7, 9, 3, 4, 5, 10, 8]
    # quick_sort2(arr, 0, len(arr) -1)
    # print(arr)
    # print(min_revolved([3, 4, 5, 6, 7, 8, 0, 1, 2]))
    # print(min_revolved([3, 4, 5, 6, 7, 8]))
    # print(min_revolved([0, 1, 1, 1, 0, 1]))

    # print(has_path([['a', 'b', 'c', 'd'],
    #                 ['e', 'f', 'g', 'h'],
    #                 ['i', 'j', 'k', 'l']],
    #                'bfgk'))
    # print(has_path([['a', 'b', 'c', 'd'],
    #                 ['e', 'f', 'g', 'h'],
    #                 ['i', 'j', 'k', 'l']],
    #                'bfgl'))
    # print(robot_moving_count(2, 3, 3))

    # print(max_product_after_cut(8))

    print(number_of1(8))