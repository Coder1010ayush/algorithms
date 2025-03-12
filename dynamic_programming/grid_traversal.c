// ----------------------------- utf-8 encoding ---------------------------------
// problem statement : let us assume we have a (m , n) grid
// how many way you have to go at a given point from a starting coordinates
//       grid of shape (3 , 3)
//      a(0,0)       b(0,1)       c(0,2)
//      d(1,0)       e(1,1)       f(1,2)
//      g(2,0)       h(2,1)       i(2,2)
//
//  if we are at point a(0,0) and we have to go to point i and possible paths can be :
//          a -> b -> c -> f -> i
//          a -> d -> g -> h -> i
//          a -> b -> e -> h -> i
//          a -> b -> e -> f -> i
//          a -> d -> e -> h -> i
//          a -> d -> e -> f -> i
// the cost of these path is same and path is covered by going right or down either or both
// of possible ways to reach at a point
// Total possible path will be (m+n)C(n)
#include <stdio.h>
#define MAXX 100
#define MAXY 100
int memo[MAXX][MAXY];

int calculate_total_number_ways(int x, int y, int x_d, int y_d)
{
    if ((x == x_d) && (y == y_d))
        return 1;

    if (x > x_d || y > y_d)
        return 0;

    return calculate_total_number_ways(x + 1, y, x_d, y_d) + calculate_total_number_ways(x, y + 1, x_d, y_d);
}

int calculate_total_number_ways_memo(int x, int y, int x_d, int y_d)
{
    if ((x == x_d) && (y == y_d))
        return 1;
    if ((x > x_d) || (y > y_d))
        return 0;
    if (memo[x][y] != -1)
        return memo[x][y];
    memo[x][y] = calculate_total_number_ways(x + 1, y, x_d, y_d) + calculate_total_number_ways(x, y + 1, x_d, y_d);
    return memo[x][y];
}
/*
    now let us add obstacle in some of cells in the given grid
    a(0,0)       b(0,1)       c(0,2)
    d(1,0)       e(1,1)       f(1,2)
    g(2,0)       h(2,1)       i(2,2)

    let us assume there is an obstacle at point e now let us figure out all possible path
    a -> b -> c -> f -> i
    a -> d -> g -> h -> i
    a -> b -> e -> Not valid , similarly other will also not be valid.
*/

int calculate_total_number_ways_obstacles(int x, int y, int x_d, int y_d, int obstacle_grid[MAXX][MAXY])
{

    // obstacle_grid[x][y] = 0 than it means there is no obstacle path is free
    // obstacle_grid[x][y] = 1 than it means there is obstacle on path so we can not go through this path
    if ((x == x_d) && (y == y_d) && obstacle_grid[x][y] == 0)
        return 1;
    if ((x > x_d) || (y > y_d) || obstacle_grid[x][y] == 1)
        return 0;

    return calculate_total_number_ways_obstacles(x + 1, y, x_d, y_d, obstacle_grid) + calculate_total_number_ways_obstacles(x, y + 1, x_d, y_d, obstacle_grid);
}

int calculate_total_number_ways_obstacles_memo(int x, int y, int x_d, int y_d, int obstacle_grid[MAXX][MAXY])
{

    // obstacle_grid[x][y] = 0 than it means there is no obstacle path is free
    // obstacle_grid[x][y] = 1 than it means there is obstacle on path so we can not go through this path
    if ((x == x_d) && (y == y_d) && obstacle_grid[x][y] == 0)
        return 1;
    if ((x > x_d) || (y > y_d) || obstacle_grid[x][y] == 1)
        return 0;

    memo[x][y] = calculate_total_number_ways_obstacles(x + 1, y, x_d, y_d, obstacle_grid) + calculate_total_number_ways_obstacles(x, y + 1, x_d, y_d, obstacle_grid);
    return memo[x][y];
}

int main()
{
    int x_d = 2;
    int y_d = 2;
    for (int i = 0; i < x_d; i++)
    {
        for (int j = 0; j < y_d; j++)
        {
            memo[i][j] = -1;
        }
    }
    int obstacle_grid[MAXX][MAXY];
    for (int i = 0; i < MAXX; i++)
    {
        for (int j = 0; j < MAXY; j++)
        {
            if (i == 1 && j == 1)
                obstacle_grid[i][j] = 1;
            else
                obstacle_grid[i][j] = 0;
        }
    }

    // int cal = calculate_total_number_ways(0, 0, 2, 2);
    int cal = calculate_total_number_ways_obstacles(0, 0, 2, 2, obstacle_grid);
    printf("Total Possible paths is %d", cal);
    return 0;
}
