// ------------------------------- utf-8 encoding ------------------------------------------------
/*
        Problem Statement : You are given an array and task is to find out that is it possible to
        find out two subset with equal sum of total sum.

        For example :
        01. Input: nums = [1, 5, 11, 5]
            Total sum = 1 + 5 + 11 + 5 = 22 (even)
            Target subset sum = 22 / 2 = 11

            Possible subsets:
            - {11} and {1, 5, 5} → both have sum = 11
            Output: true

        02. Input: nums = [1, 2, 3, 5]
            Total sum = 1 + 2 + 3 + 5 = 11 (odd)
            Impossible to split into two equal subsets.
            Output: false

        03. Input: nums = [1, 2, 3, 4, 6]
            Total sum = 1 + 2 + 3 + 4 + 6 = 16 (even)
            Target subset sum = 16 / 2 = 8

            Possible subsets:
            - {2, 6} → sum = 8
            - {1, 3, 4} → sum = 8
            Output: true

*/
#include <stdio.h>
#define Min_x 100
#define Max_x 100
int memo[Min_x][Max_x];
int helper(int array[], int target, int index)
{
    // target is equal to sum of subset
    if (target == 0)
        return 1;
    // array is now empty
    if (index == 0)
        return 0;

    // if the element is greater than the target than we can skip this (obvious reason)
    if (target < array[index - 1])
        return helper(array, target, index - 1);

    // take condition
    int take = helper(array, target - array[index - 1], index - 1);
    // skip condition
    int skip = helper(array, target, index - 1);

    // take the or of both condition
    return take || skip;
}
int helper_memo(int array[], int target, int index)
{
    // target is equal to sum of subset
    if (target == 0)
        return 1;
    // array is now empty
    if (index == 0)
        return 0;
    if (memo[index][target] != -1)
    {
        return memo[index][target];
    }

    // if the element is greater than the target than we can skip this (obvious reason)
    if (target < array[index - 1])
        return memo[index][target] = helper(array, target, index - 1);

    // take condition
    int take = helper(array, target - array[index - 1], index - 1);
    // skip condition
    int skip = helper(array, target, index - 1);

    // take the or of both condition
    return memo[index][target] = (take || skip);
}
int find_equal_subset(int array[], int size)
{
    int total_sum = 0;
    for (int i = 0; i < size; i++)
    {
        total_sum += array[i];
    }

    // total_sum is odd than there can be equal sum of two subset
    if (total_sum % 2 != 0)
    {
        printf("total sum is %d and total size is %d \n ", total_sum, size + 1);
        return 0;
    }

    // now  our target is to find out is there any subset present with sum of total_sun / 2
    int res = helper(array, total_sum / 2, size);
    return res;
}
int main()
{
    int array[] = {1, 2, 3, 4, 6};
    int size = 5;

    // 0 means false , 1 means true
    int res = find_equal_subset(array, size);
    printf("Output is %d \n ", res);
    return 0;
}