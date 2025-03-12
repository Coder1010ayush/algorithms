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
int main()
{
    for (int i = 0; i < MAX; i++)
    {
        memo[i] = -1;
    }
    int number_of_stairs = 4;
    // int res = calculate_number_ways_climb_stairs(4);
    int res = calculate_number_ways_climb_stairs_memo(4);
    printf("Total number of ways is %d ", res);
}