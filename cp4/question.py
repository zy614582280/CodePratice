# -- coding: utf-8 --
from collections import deque
from typing import List


class BinaryTreeNode():
    def __init__(self,
                 root,
                 left=None,
                 right=None):
        self.root = root
        self.left = left
        self.right = right


def mirror_binary_tree(p_node: BinaryTreeNode):
    """
    输入一个二叉树，输出它的镜像
    :param p_node:
    :return:
    """
    # 如果当前节点为空、或者是叶子节点，则不交换
    if not p_node or (p_node.left is None and p_node.right is None):
        return
    # 交换左右节点
    tmp = p_node.left
    p_node.left = p_node.right
    p_node.right = tmp
    # 递归
    mirror_binary_tree(p_node.left)
    mirror_binary_tree(p_node.right)


def mirror_binary_tree2(p_node: BinaryTreeNode):
    """
    前序遍历，输出二叉树的镜像
    :param p_node:
    :return:
    """
    if not p_node or (p_node.left is None and p_node.right is None):
        return
    stack = [p_node]
    while len(stack) > 0:
        node = stack.pop()
        if node.left or node.right:
            tmp = node.left
            node.left = node.right
            node.right = tmp
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
    print_binary_tree(p_node)


def print_binary_tree(p_node: BinaryTreeNode):
    """
    分行从上到下打印二叉树
    :param p_node:
    :return:
    """
    if not p_node:
        return
    queue = deque()
    queue.append("r")
    queue.append(p_node)
    while len(queue) > 0:
        node = queue.popleft()
        if node == "r":
            if len(queue) > 0:
                queue.append("r")
                print()
        else:
            print(node.root, end=" ")
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    print()


def is_mirror_tree(p_node1: BinaryTreeNode,
                   p_node2: BinaryTreeNode):
    """
    判断一个二叉树是否是镜像对称二叉树
    如果一个二叉树是一个对称二叉树，那么满足
    left.right == right.left
    left_.left == right.right
    :param p_node1:
    :param p_node2:
    :return:
    """
    if not p_node1 and not p_node2:
        return True
    if not p_node1 or not p_node2:
        return False
    if p_node1.root != p_node2.root:
        return False
    return is_mirror_tree(p_node1.left, p_node2.right) and is_mirror_tree(p_node1.right, p_node2.left)


def print_matrix_in_circle(matrix: List[List[int]]):
    if not matrix:
        return []
    result = []
    matrix_copy = matrix
    while matrix_copy:
        # 弹出第一行
        result += matrix_copy.pop(0)
        # 逆时针旋转90°
        if matrix_copy:
            matrix_copy = turn(matrix_copy)
    return result


def turn(matrix: List[List[int]]):
    """
    逆时针90°翻转矩阵
    [[1, 2, 3],       [[3, 6],
     [4, 5, 6]]   ->   [2, 5],
    :param matrix:     [1, 4]]
    :return:
    """
    row = len(matrix)
    col = len(matrix[0])
    new_matrix = []
    for c in range(col-1, -1, -1):
        new_row = []
        for r in range(row):
            new_row.append(matrix[r][c])
        new_matrix.append(new_row)
    return new_matrix


class StackWithMin():
    """
    包含min函数的栈
    定义栈的数据结构，包含 push、pop、min方法，且时间复杂度都是O(1)。
    """
    def __init__(self):
        self.data = []
        self.min_values = []

    def push(self, value):
        if not self.data:
            self.min_values.append(value)
        else:
            min_value = min(self.min_values[-1], value)
            self.min_values.append(min_value)
        self.data.append(value)

    def pop(self):
        if not self.data:
            return None
        self.min_values.pop(0)
        return self.data.pop(0)

    def min(self):
        if not self.data or not self.min_values:
            return None
        return self.min_values[-1]


def is_pop_order(push_seq: List[int],
                 pop_seq: List[int]):
    """
    压入、出栈 顺序
    :param push_seq:
    :param pop_seq:
    :return:
    """
    if not push_seq and not pop_seq:
        return True
    if not push_seq or not pop_seq:
        return False
    # 辅助栈
    stack = []
    i, j = 0, 0  # 分别指向push_seq、pop_seq
    while j < len(pop_seq):
        # 如果当前栈为空，而且栈顶元素不等于pop_seq当前元素
        if not stack or stack[-1] != pop_seq[j]:
            # 如果push_seq还有元素没有入栈，则压栈
            if i < len(push_seq):
                stack.append(push_seq[i])
                i += 1
            else:
                # 如果push_seq元素都入栈了，则打断，这里其实就可以返回False了
                return False
        elif stack[-1] == pop_seq[j]:
            # 如果栈顶元素等于pop_seq当前元素，则出栈
            stack.pop()
            j += 1

    return True


def print_tree_from_top_to_bottom(p_node: BinaryTreeNode):
    """
    不分行从上到下打印二叉树
    树的广度优先遍历（BFS），用队列实现
    :param p_node:
    :return:
    """
    if not p_node:
        return []

    result = []
    queue = deque()
    queue.append(p_node)
    while queue:
        node = queue.popleft()
        result.append(node.root)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return result


def print_tree_by_row(p_root: BinaryTreeNode):
    """
    分行打印二叉树
    还是树的广度优先遍历(BFS)，由于分行打印，需要新增两个变量：当前行未打印元素数，下一行元素数
    :param p_root:
    :return:
    """
    if not p_root:
        return
    queue = deque()
    queue.append(p_root)
    current_row_num = 1
    next_row_num = 0
    while queue:
        p_node = queue.popleft()
        current_row_num -= 1
        print(p_node.root, end=" ")
        if p_node.left:
            queue.append(p_node.left)
            next_row_num += 1
        if p_node.right:
            queue.append(p_node.right)
            next_row_num += 1

        if current_row_num == 0:
            print()
            current_row_num = next_row_num
            next_row_num = 0


def print_tree_by_s(p_root: BinaryTreeNode):
    """
    之字型、S型 打印二叉树
    第一行：从左向右
    第二行：从右向左
    第三行：从左向右
    ...
    :param p_root:
    :return:
    """
    if not p_root:
        return
    stack1, stack2 = [], []  # stack1为奇数行栈，stack2为偶数行栈
    stack1.append(p_root)
    while stack1 or stack2:
        if stack1:
            while stack1:
                p_node = stack1.pop()
                print(p_node.root, end=" ")
                # 奇数栈，先进左子树、再进右子树
                if p_node.left:
                    stack2.append(p_node.left)
                if p_node.right:
                    stack2.append(p_node.right)
            print()
        if stack2:
            while stack2:
                p_node = stack2.pop()
                print(p_node.root, end=" ")
                # 偶数栈，先进右子树，再进左子树
                if p_node.right:
                    stack1.append(p_node.right)
                if p_node.left:
                    stack1.append(p_node.left)
            print()


if __name__ == '__main__':
    root, l1, r1, l2, r2, l3, r3 = BinaryTreeNode(8), BinaryTreeNode(6), BinaryTreeNode(7), BinaryTreeNode(5), \
                                   BinaryTreeNode(7),  BinaryTreeNode(7),  BinaryTreeNode(1)
    root.left = l1
    root.right = r1
    l1.left = l2
    l1.right = r2
    r1.left = l3
    r1.right = r3

    print_binary_tree(root)

    print(print_tree_from_top_to_bottom(root))
    print_tree_by_row(root)
    print_tree_by_s(root)

    # mirror_binary_tree2(root)
    # print(is_mirror_tree(root, root))

    # matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # print(print_matrix_in_circle(matrix))

    # stack = StackWithMin()
    # print("push 1")
    # stack.push(1)
    # print("min=", stack.min())
    # stack.push(2)
    # print("push 2")
    # print("min=", stack.min())
    # print("push 0")
    # stack.push(0)
    # print("min=", stack.min())
    # print("pop ", stack.pop())
    # print("min=", stack.min())

    # print(is_pop_order(push_seq=[1, 2, 3, 4, 5],
    #                    pop_seq=[4, 5, 3, 2, 1]))
    #
    # print(is_pop_order(push_seq=[1, 2, 3, 4, 5],
    #                    pop_seq=[4, 3, 5, 1, 2]))
