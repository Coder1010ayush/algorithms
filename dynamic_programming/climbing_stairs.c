// --------------------------------------------------- utf-8 encoding ----------------------------------------
/*

    Climbing Stairs Problem :
        You can take 1 or 2 steps at a time. Find the number of distinct ways to reach the nth step.
            Example:
                Input: n = 3
                Output: 3 (Ways: [1,1,1], [1,2], [2,1])

                Input: n = 4
                Output: 5 (Ways: [1,1,1,1], [1,1,2],[2,2],[1,2,1],[2,1,1])
*/
#include <stdio.h>
#define MAX 100
int memo[MAX];

int calculate_number_ways_climb_stairs(int n)
{
    if (n == 2)
        return 2;
    if (n == 1)
        return 1;
    return calculate_number_ways_climb_stairs(n - 1) + calculate_number_ways_climb_stairs(n - 2);
}

int calculate_number_ways_climb_stairs_memo(int n)
{
    if (n == 2)
        return 2;
    if (n == 1)
        return 1;
    if (memo[n] != -1)
        return memo[n];
    memo[n] = calculate_number_ways_climb_stairs_memo(n - 1) + calculate_number_ways_climb_stairs_memo(n - 2);
    return memo[n];
}

/*
    let us generalize this problem instead of taking 1 or 2 steps , steps will be taken from  a list or set given.
*/
int calculate_number_ways_climb_stairs_general(int n, int steps[], int step_count)
{

    if (n == 0)
        return 1;
    if (n < 0)
        return 0;

    int t = 0;
    for (int i = 0; i < step_count; i++)
    {
        t += calculate_number_ways_climb_stairs_general(n - steps[i], steps, step_count);
    }
    return t;
}

int calculate_number_ways_climb_stairs_general_memo(int n, int steps[], int step_count)
{

    if (n == 0)
        return 1;
    if (n < 0)
        return 0;
    if (memo[n] != -1)
        return memo[n];

    int t = 0;
    for (int i = 0; i < step_count; i++)
    {
        t += calculate_number_ways_climb_stairs_general(n - steps[i], steps, step_count);
    }
    memo[n] = t;
    return memo[n];
}

int main()
{
    for (int i = 0; i < MAX; i++)
    {
        memo[i] = -1;
    }
    int number_of_stairs = 4;
    int steps[2] = {1, 2};
    int step_count = 2;
    // int res = calculate_number_ways_climb_stairs(4);
    // int res = calculate_number_ways_climb_stairs_memo(4);
    int res = calculate_number_ways_climb_stairs_general(number_of_stairs, steps, step_count);
    printf("Total number of ways is %d ", res);
}