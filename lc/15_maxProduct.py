# -- coding: utf-8 --

def maxProduct(nums):
    """
    :type nums: List[int]
    :rtype: int
    乘积最大子数组
    """
    if not nums:
        return 0
    # 最大值，包含元素nums[i]的最大值，包含元素nums[i]的最小值
    # 保存最大值、最小值的原因:
    # 举个例子， nums[i] = -1, cur_max = 2, cur_min = -8
    # 那么包含nums[i]的最大值是 nums[i] * cur_min，包含nums[i]的最小值是 nums[i] * cur_max
    # 会出现这种两极翻转的情况，所以需要把包含nums[i]的最大值、最小值都保留下来
    max_value, cur_max, cur_min = nums[0], nums[0], nums[0]
    for num in nums[1:]:
        pre_max, pre_min = cur_max, cur_min
        cur_max = max(num, num * pre_max, num * pre_min)
        cur_min = min(num, num * pre_max, num * pre_min)
        max_value = max(cur_max, max_value)
    return max_value


def maxProduct2(nums):
    """
    动态规划
    dp_max[i]表示包含num[i]的最大值
    dp_min[i]表示包含num[i]的最小值
    与其他动态规划不同的在于，这里dp_max[i]只依赖dp_max[i-1]、dp_min[i-1]
    i-1之前的元素完全不需要记录，所以dp_max、dp_min 可以用两个变量代替，而不是用列表
    就变成了方法1了
    :param nums:
    :return:
    """
    if not nums:
        return 0
    dp_max = [nums[0]]
    dp_min = [nums[0]]
    max_value = dp_max[0]
    for num in nums[1:]:
        pre_max, pre_min = dp_max[-1], dp_min[-1]
        dp_max.append(max(pre_max * num, pre_min * num, num))
        dp_min.append(max(pre_max * num, pre_min * num, num))
        max_value = max(max_value, dp_max[-1])
    return max_value


if __name__ == '__main__':
    print(maxProduct([2,3,-2,4]))
    print(maxProduct([-2,0,-1]))
    print(maxProduct([-2,-2,-1]))

    print(maxProduct2([2,3,-2,4]))
    print(maxProduct2([-2,0,-1]))
    print(maxProduct2([-2,-2,-1]))