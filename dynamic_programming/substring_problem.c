// ----------------------------------- utf-8 encoding ---------------------------------------
// header files and macros used in this file
#include <stdio.h>
#include <string.h>
#define Min_X 100
#define Max_X 100
/*
    Problem Statement : Return True if a given string is palindromic in nature otherwise return False.asm
    Pallindrom String : A string that reads the same from forward as backward.
    Test Cases :
        01. Input : "aba"
            Ouput : True / 1
        02. Input : "abc"
            Output : False / 0

*/

int check_pallindromic_nature(char *c)
{
    int length = strlen(c);
    int mid = length / 2;

    /*
        there could be two cases whether length of string could be even or odd.
        if even and mid and mid - 1 both index character will be same.
        if odd than mid index character will be unique.
        condition [ index and length - index character will be same ]
    */
    int res = 1; // initially let us assume string is pallindromic in nature.
    for (int i = 0; i <= mid - 1; i++)
    {
        char f = c[i];
        char s = c[length - 1 - i];
        if (f != s)
        {
            res = 0;
            break;
        }
    }
    return res;
}
/*
    Problem Statement : Given a string s find the longest palindromic substring in it.
    A palindrome is a string that reads the same backward as forward.
    Test Cases :
        01. Input: "babad"
            Output: "bab" or "aba"

        02. Input: "cbbd"
            Output: "bb"

        03. Input: "forgeeksskeegfor"
            Output: "geeksskeeg"

        04. Input: "abcda"
            Output: "a"

        05. Input: "a"
            Output: "a"
*/
// function for finding longest pallindromic substring in a given string.
char *find_pallindromic_string_longest(char *str, int index)
{
}

// main function
int main(void)
{
    char s[] = "abba";
    int res = check_pallindromic_nature(s);
    printf("String is pallindromic in nature %d \n", res);
    return 0;
}