// ---------------------------------------------------- utf-8 encoding* --------------------------
/*

    Coin Change(Ways to Make Change)
        Given an array of coin denominations and an amount,
        find the number of ways to make the amount.
            Example : Input : coins = [ 1, 2, 5 ], amount = 5
                      Output : 4 Ways : {5}, {2, 2, 1}, {2, 1, 1, 1}, {1, 1, 1, 1, 1}

*/
#include <stdio.h>
#include <string.h>
#define MAX_X 100
int memo[MAX_X];

int count_coine(int amount, int coins[], int coint_count)
{
    if (amount == 1)
        return 1;
    if (amount <= 0)
        return 0;

    int ways = 0;
    for (int i = 0; i < coint_count; i++)
    {
        ways += count_coine(amount - coins[i], coins, coint_count);
    }
    return ways;
}
int count_coine_memo(int amount, int coins[], int coint_count)
{
    if (amount == 1)
        return 1;
    if (amount <= 0)
        return 0;
    if (memo[amount] != -1)
        return memo[amount];

    int ways = 0;
    for (int i = 0; i < coint_count; i++)
    {
        ways += count_coine_memo(amount - coins[i], coins, coint_count);
    }
    memo[amount] = ways;
    return memo[amount];
}

int main()
{
    for (int i = 0; i < MAX_X; i++)
    {
        memo[i] = -1;
    }
    int amount = 5;
    int coins[] = {1, 2};
    int count = 2;
    int res = count_coine(amount, coins, count);
    printf("Total number of ways to make amount using given coins is %d ", res);
}
