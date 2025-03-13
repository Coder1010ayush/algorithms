// ------------------------------------------------- *utf-8 encoding* -------------------------------------------------
/*
    Jump Game
        Given an array where each element represents maximum jump length, determine if you can reach the last index.
            Example:
                Input: nums = [2,3,1,1,4]
                Output: true (Jump 2 → 3 → 4)

                Input: nums = [3,2,1,0,4]
                Output: false

                Input: nums = [2,3,1,1,0,0,0,4]
                Output: false

                Input: nums = [5,1,1,1,1,1]
                Output: true

*/
#include <stdio.h>
#define MAX_X 100
int memo[MAX_X];

int find_can_be_reached(int number_of_steps, int steps[], int index)
{
    if (index >= number_of_steps)
        return 0;
    if (index == number_of_steps - 1)
        return 1;
    if (steps[index] == 0)
        return 0;

    int ways = steps[index];
    int val = 0;
    for (int i = 1; i <= ways; i++)
    {
        val = val || find_can_be_reached(number_of_steps, steps, index + i);
    }
    return val;
}

int find_can_be_reached_memo(int number_of_steps, int steps[], int index)
{
    if (index >= number_of_steps)
        return 0;
    if (index == number_of_steps - 1)
        return 1;
    if (steps[index] == 0)
        return 0;

    if (memo[index] != -1)
        return memo[index];

    int ways = steps[index];
    int val = 0;
    for (int i = 1; i <= ways; i++)
    {
        val = val || find_can_be_reached(number_of_steps, steps, index + i);
    }
    return memo[index] = val;
}

int main()
{
    for (int i = 0; i < MAX_X; i++)
        memo[i] = -1;

    int steps[8] = {2, 3, 1, 1, 0, 0, 0, 4};
    // int steps[6] = {5, 1, 1, 1, 1, 1};
    // int number_of_steps = 6;
    int number_of_steps = 8;
    // int res = find_can_be_reached(number_of_steps, steps, 0);
    int res = find_can_be_reached_memo(number_of_steps, steps, 0);
    printf("Can be reached %d ", res);
    return 0;
}