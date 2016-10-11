#include <iostream>
#include <stdio.h>
#include <cstdlib>
#include <omp.h>

#define N 200

//using namespace std;

int main(int argc, const char** argv)
{

  int n = BLOCK_SIZE * (N/BLOCK_SIZE);
  int a[N][N];
  int b[N][N];
  int c[N][N];
  int sum=0;

  //omp_set_dynamic(0); 
  omp_set_num_threads(4);
  //cout<<"threads="<<omp_get_num_threads()<<endl;

  #pragma omp parallel for 
  for(int k1=0;k1<n;k1+=BLOCK_SIZE)
  {
      //cout<<"threads="<<omp_get_num_threads()<<endl;
      for(int j1=0;j1<n;j1+=BLOCK_SIZE)
      {
          for(int k1=0;k1<n;k1+=BLOCK_SIZE)
          {
              for(int i=0;i<n;i++)
              {
                  for(int j=j1;j<j1+BLOCK_SIZE;j++)
                  {
                      sum = c[i][j];
                      for(int k=k1;k<k1+BLOCK_SIZE;k++)
                      {               
                          sum += a[i][k] * b[k][j];
                      }
                      c[i][j] = sum;
                  }
              }
          }
      }
         }
  return 0;
}
