# -- coding: utf-8 --
import re
from typing import List, Optional


def power_with_exponent(base,
                        exponent):
    """
    数值的整数次方
    实现函数power(base, exponent)，求base的exponent次方。
    1. 考虑 exponent 的正负问题
    2. 通过递归计算次方运算，降低时间复杂度
    :param base:
    :param exponent:
    :return:
    """
    if base == 0 and exponent <= 0:
        return 0
    if exponent == 0:
        return 1
    if exponent == 1:
        return base
    negative = False
    if exponent < 0:
        exponent = abs(exponent)
        negative = True
    """
    这里用递归来实现
    2**4 = (2**2) * (2**2)
    2**5= (2**2) * (2**2) * 2
    """
    result = pow_with_exponent_core(base, exponent)
    if negative:
        result = 1 / result
    return result


def pow_with_exponent_core(base,
                           exponent):
    """
    递归计算n次方，时间复杂度O(log(n))
    :param base:
    :param exponent:
    :return:
    """
    if exponent == 0:
        return 1
    if exponent == 1:
        return base
    result = pow_with_exponent_core(base,
                                    exponent // 2)
    result *= result
    if exponent % 2 == 1:
        result *= base
    return result


def print_1_to_max(n: int):
    """
    打印从1到最大的n位数
    需要考虑大数的情况，用字符串或者数组代替整数类型
    :param n:
    :return:
    """
    pass


class ListNode:
    def __init__(self,
                 value,
                 p_next=None):
        self.value = value
        self.p_next = p_next


def delete_node(p_list_head: ListNode,
                p_to_be_deleted: ListNode):
    """
    删除链表的节点，时间复杂度要求在O(1)
    :param p_list_head:
    :param p_to_be_deleted:
    :return:
    """
    if p_list_head is None or p_to_be_deleted is None:
        return
    if p_to_be_deleted.p_next is not None:
        # 如果删除的节点有后续节点
        p_to_be_deleted.value = p_to_be_deleted.p_next.value
        p_to_be_deleted.p_next = p_to_be_deleted.p_next.p_next
    elif p_list_head.value == p_to_be_deleted.value:
        # 如果删除的节点是最后一个节点，也是第一个节点
        p_list_head = None
        p_to_be_deleted = None
    else:
        # 如果删除的节点是最后一个节点，则从头到尾串行删除
        while p_list_head.p_next != p_to_be_deleted.value:
            p_list_head.value = p_list_head.p_next.value
            p_list_head.p_next = p_list_head.p_next.p_next


def delete_duplicate_node(p_list_head: ListNode):
    """
    删除链表中重复的元素
    输入：1 -> 2 -> 2 -> 2 -> 3 -> 3 -> 4
    输出：1 -> 4
    需要考虑一点，头节点可能是重复元素
    :param p_list_head:
    :return:
    """
    if p_list_head is None or p_list_head.p_next is None:
        return p_list_head

    first_p_node = ListNode(-1, p_list_head)  # 新建的首节点
    last_p_node = first_p_node.p_next  # 上一个不同的节点
    while p_list_head and p_list_head.p_next:
        if p_list_head.value == p_list_head.p_next.value:
            # 如果当前节点与下一个节点的值相同，那么继续遍历下一个节点，直至找到一个不同值的节点
            duplicate_value = p_list_head.value
            while p_list_head and p_list_head.value == duplicate_value:
                p_list_head = p_list_head.p_next
            # 将last node 的下一个节点指向不同值的节点
            last_p_node.p_next = p_list_head
        else:
            # 如果不等，则p_list_head、last_p_node 右移一位
            last_p_node = p_list_head
            p_list_head = p_list_head.p_next

    return first_p_node.p_next


def regex_match(text,
                pattern):
    if not text or not pattern:
        return False
    return regex_match_core(text, pattern)


def regex_match_core(text: str,
                     pattern: str
                     ):
    """
    1. text、pattern均为空，返回True
    2. text不为空，pattern为空，返回False
    3. text为空，pattern不为空，如果pattern = (ch)* 则返回True
    4. text、pattern 都不为空
    :param text:
    :param pattern:
    :return:
    """
    if not text:
        if not pattern or (len(pattern) == 2 and pattern[1] == '*'):
            return True
        return False
    if not pattern:
        return False
    # 如果pattern是 (ch)* 格式，则有三种匹配方式
    # 1. pattern右移两位，text不动
    # 2. pattern不动，text右移一位
    # 3. pattern右移两位，text右移一位
    if len(pattern) >= 2 and pattern[1] == '*':
        if text[0] == pattern[0] or pattern[0] == '.':
            return regex_match_core(text[1:], pattern) or regex_match_core(text[1:], pattern[2:])
        else:
            return regex_match_core(text, pattern[2:])
    if pattern[0] == text[0] or pattern[0] == '.':
        # 第二种情况，pattern、text均右移一位
        return regex_match_core(text[1:], pattern[1:])
    return False


def is_number(text: str) -> bool:
    """
    判断字符串是否表示数值，包括整数、小数、科学计数法
    +100、5e2、-123、3.14、-1E-16:True
    12e、
    :param text:
    :return:

    数值遵循 A[.[B]][e|EC] 或者 .B[e|EC]
    """
    import re
    pattern1 = re.compile(r'[+-]?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?')
    pattern2 = re.compile(r'\.[0-9]+([eE][+-]?[0-9]+)?')
    pattern3 = re.compile(r'[+-]?[0-9]+\.')
    if pattern1.fullmatch(text) or pattern2.fullmatch(text) or pattern3.fullmatch(text):
        return True
    return False


def reorder_odd_even(nums: List[int]):
    if not nums:
        return
    p1 = 0
    p2 = len(nums) - 1
    while p1 <= p2:
        while p1 < len(nums) and nums[p1] % 2 == 1:
            p1 += 1
        while p2 >= 0 and nums[p2] % 2 == 0:
            p2 -= 1

        # 加这个限制条件，是为了防止最后一次位移后，p2在p1左边
        if p1 < p2:
            tmp = nums[p1]
            nums[p1] = nums[p2]
            nums[p2] = tmp


def find_kth_to_tail(p_list_head: ListNode,
                     k_th: int):
    """
    找到链表倒数第k个元素
    :param p_list_head: 头节点
    :return:
    """
    if p_list_head is None or k_th <= 0:
        return None

    p1 = p_list_head
    p2 = p_list_head
    # p1 向前移动k-1步
    for i in range(k_th - 1):
        if p1.p_next is not None:
            p1 = p1.p_next
        else:
            return None

    while p1.p_next is not None:
        p1 = p1.p_next
        p2 = p2.p_next
    return p2


def find_entry_node_of_loop(p_list_head: ListNode):
    """
    找到链表中环的入口节点
    :param p_list_head:
    :return:
    """
    # 1. 判断是否有环，找出环内节点
    meet_node = meeting_node(p_list_head)
    if not meet_node:
        return None
    # 2. 算出环内有几个节点
    tag_node = meet_node
    count = 1
    while meet_node.p_next != tag_node:
        count += 1
        tag_node = tag_node.p_next
    # 3. 快慢指针，找出入环节点
    p_fast = p_list_head
    p_show = p_list_head
    # p_fast 先移动count步
    for i in range(count):
        p_fast = p_fast.p_next
    # p_fast p_slow 同步移动，两者相遇
    while p_show != p_fast:
        p_fast = p_fast.p_next
        p_show = p_show.p_next
    return p_show


def meeting_node(p_list_head: ListNode) -> Optional[ListNode]:
    if p_list_head is None or p_list_head.p_next is None:
        return None
    # 判断是否存在环
    p_slow = p_list_head
    p_fast = p_slow.p_next
    while p_slow and p_fast:
        if p_fast == p_slow:
            return p_fast
        # p_slow 走一步
        p_slow = p_slow.p_next

        # p_fast 走两步
        if p_slow.p_next:
            p_fast = p_slow.p_next

    return None


def reverse_list(p_list_head: ListNode):
    if p_list_head is None or p_list_head.p_next is None:
        return p_list_head

    p_pre = None
    p_next = None
    while p_list_head is not None:
        p_next = p_list_head.p_next  # 先保存下一个节点
        p_list_head.p_next = p_pre  # 将当前节点的下一个节点执行前面

        p_pre = p_list_head  # 已反转链表 + 当前节点
        p_list_head = p_next
    return p_pre


def merge_sorted_list(p_list_head1: ListNode,
                      p_list_head2: ListNode):
    if not p_list_head1:
        return p_list_head2
    if not p_list_head2:
        return p_list_head1

    p_merged = ListNode(-1)
    p_merged_head = p_merged
    while p_list_head1 and p_list_head2:
        if p_list_head1.value <= p_list_head2.value:
            p_merged.p_next = p_list_head1
            p_list_head1 = p_list_head1.p_next
        else:
            p_merged.p_next = p_list_head2
            p_list_head2 = p_list_head2.p_next
        p_merged = p_merged.p_next
    if p_list_head1:
        p_merged.p_next = p_list_head1
    if p_list_head2:
        p_merged.p_next = p_list_head2

    return p_merged_head.p_next


def merge_sorted_list_recursive(p_list_head1: ListNode,
                                p_list_head2: ListNode):
    if not p_list_head1:
        return p_list_head2
    if not p_list_head2:
        return p_list_head1
    if p_list_head1.value <= p_list_head2.value:
        p_list_head1.p_next = merge_sorted_list_recursive(p_list_head1.p_next, p_list_head2)
        return p_list_head1
    else:
        p_list_head2.p_next = merge_sorted_list_recursive(p_list_head1, p_list_head2.p_next)
        return p_list_head2


def print_list_node(p_list_head: ListNode):
    if not p_list_head:
        return
    p_node = p_list_head
    while p_node:
        print(p_node.value)
        p_node = p_node.p_next


class BinaryTreeNode:
    def __init__(self,
                 root: int,
                 left=None,
                 right=None):
        self.root = root
        self.left = left
        self.right = right


def contains_tree_node(tree_node1: BinaryTreeNode,
                       tree_node2: BinaryTreeNode):
    if not tree_node1:
        return False
    if not tree_node2:
        return True
    if tree_node1.root != tree_node2.root:
        return False
    return contains_tree_node(tree_node1.left, tree_node2.left) \
           and contains_tree_node(tree_node1.right, tree_node2.right)


def has_sub_tree(tree_node1: BinaryTreeNode,
                 tree_node2: BinaryTreeNode):
    result = False
    if tree_node1 and tree_node2:
        result = contains_tree_node(tree_node1, tree_node2)
        if not result:
            result = has_sub_tree(tree_node1.left, tree_node2)
        if not result:
            result = has_sub_tree(tree_node1.right, tree_node2)
    return result


if __name__ == '__main__':
    # print(power_with_exponent(6, 6))

    # print(regex_match('ab', 'abb*'))

    # print(is_number('+100'))
    # print(is_number('3.14'))
    # print(is_number('-1E-16'))
    # print(is_number('12e'))
    # print(is_number('12.'))

    # nums = [1, 2, 3, 4, 5]
    # reorder_odd_even(nums=nums)
    # print(nums)
    p1 = ListNode(1)
    p2 = ListNode(3)
    p3 = ListNode(5)
    p1.p_next = p2
    p2.p_next = p3

    p4 = ListNode(2)
    p5 = ListNode(4)
    p6 = ListNode(6)
    p4.p_next = p5
    p5.p_next = p6

    # reversed_list = reverse_list(p1)
    # while reversed_list:
    #     print(reversed_list.value)
    #     reversed_list = reversed_list.p_next
    # print(reversed_list.value, reversed_list.p_next.value, reversed_list.p_next.p_next)

    p_merged = merge_sorted_list_recursive(p1, p4)
    print_list_node(p_merged)
