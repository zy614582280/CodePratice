# -- coding: utf-8 --
"""
多数元素

给定一个大小为 n 的数组，找到其中的多数元素。多数元素是指在数组中出现次数 大于⌊n/2⌋的元素。

你可以假设数组是非空的，并且给定的数组总是存在多数元素。

示例1：
输入：[3,2,3]
输出：3

示例2：
输入：[2,2,1,1,1,2,2]
输出：2

进阶：
尝试设计时间复杂度为 O(n)、空间复杂度为 O(1) 的算法解决此问题。
"""


def majorityElement(nums):
    """
    :type nums: List[int]
    :rtype: int
    用我最喜欢的摩尔投票法
    假设数组里面都是来自不同国家的人，任意两个国家之间的人同归于尽，那么最后剩下的幸存者一定是众数
    """
    majority = nums[0]
    count = 1
    for i in range(1, len(nums)):
        if count == 0:
            majority = nums[i]
            count = 1
        elif majority == nums[i]:
            count += 1
        else:
            count -= 1
    return majority


def majorityElement2(nums):
    """
     :type nums: List[int]
     :rtype: int
     先排序，然后取中位数
     """
    nums.sort()
    return nums[len(nums) // 2]


def majorityElement3(nums):
    """
     :type nums: List[int]
     :rtype: int
     用一个计数器，这个方法太次，空间复杂度O(n)
     """
    value_count = dict()
    for num in nums:
        value_count[num] = value_count.get(num, 0) + 1
        if value_count[num] > len(nums) // 2:
            return num


if __name__ == '__main__':
    print(majorityElement([3,2,3]))
    print(majorityElement([2,2,1,1,1,2,2]))

    print(majorityElement2([3,2,3]))
    print(majorityElement2([2,2,1,1,1,2,2]))

    print(majorityElement3([3,2,3]))
    print(majorityElement3([2,2,1,1,1,2,2]))

