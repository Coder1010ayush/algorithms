//  ----------------------------------------- *utf-8 encoding* --------------------------------
// two approaches => top to down and bottom to up

//                           7
//                  6                  5
//            5           4         4      3
//        4      3     3     2    3   2  2    1
//      3   2  2   1 2   1     2    1
#include <stdio.h>
#define MAX_N 100
int memo[MAX_N];

// time complexity is O(2^n) and space complexity is O(n)
int fibo(int n)
{
    if (n <= 2) // see above diagram
        return 1;

    return fibo(n - 1) + fibo(n - 2);
}

int fibo_memo(int n)
{
    if (memo[n] != 0)
        return memo[n];

    if (n <= 2)
        return 1;
    memo[n] = fibo(n - 1) + fibo(n - 2);
}
int main()
{
    int a = 10;
    int result = fibo(10);
    printf("%d", result);
    return 0;
}