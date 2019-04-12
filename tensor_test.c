/*************************************************************************
  > File Name: tensor_test.c
  > Author:perrynzhou 
  > Mail:perrynzhou@gmail.com 
  > Created Time: Fri 12 Apr 2019 11:58:54 PM CST
 ************************************************************************/

#include <stdio.h>
#include <tensorflow/c/c_api.h>
int main(void)
{
    fprintf(stdout, "hello from tensorflow C library versiob %s\n", TF_Version());
    return 0;
}
