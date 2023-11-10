class Solution(object):
    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        for i, num in enumerate(nums):
            if type(nums[int(num) - 1]) == str:
                return int(num)
            else:
                nums[int(num) - 1] = str(nums[int(num) - 1])

sol = Solution()
sol.findDuplicate([1,3,4,2,2])