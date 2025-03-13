// --------------------------------------------------- utf-8 encoding ---------------------------------------------------
/*
    House Robber
        Given an array of house values, find the maximum sum you can rob without robbing adjacent houses.
        Example:
            Input: houses = [2,7,9,3,1]
            Output: 12 (2 + 9 + 1)

*/

#include <stdio.h>
#define MAX_X 100
int memo[MAX_X];

int find_total_cost(int number_of_house, int costs[], int index)
{
    if (index >= number_of_house)
        return 0;

    int take = costs[index] + find_total_cost(number_of_house, costs, index + 2);
    int skip = find_total_cost(number_of_house, costs, index + 1);

    if (take > skip)
        return take;
    else
        return skip;
}

int find_total_cost_memo(int number_of_house, int costs[], int index)
{

    if (index >= number_of_house)
        return 0;
    if (memo[index] != -1)
        return memo[index];

    int take = costs[index] + find_total_cost_memo(number_of_house, costs, index + 2);
    int skip = find_total_cost_memo(number_of_house, costs, index + 1);
    if (take > skip)
        return memo[index] = take;
    else
        return memo[index] = skip;
}

int main()
{
    // initializing memo array with -1 value.
    for (int i = 0; i < MAX_X; i++)
        memo[i] = -1;

    int number_of_houses = 5;
    // int costs[5] = {1, 2, 4, 5, 10};
    int costs[5] = {2, 7, 9, 3, 1};
    // int res = find_total_cost(number_of_houses, costs, 0);
    int res = find_total_cost_memo(number_of_houses, costs, 0);
    printf("Total earned profit is %d ", res);
}