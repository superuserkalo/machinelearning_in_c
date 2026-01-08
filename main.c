#include <stdio.h>
#include <math.h>
#include "dataset.h"

// Notations:
// # - number
// th - params
// x - features
// m - # training example / rows
// y - output
// n - # of feautres
// (x,y) - one training example(e.g. one row of trainign example)
// (x[i],y[i]) - one training example at a specific index
// hth(x) = for (j=0,j<=m,j++) {thj*xj}

// x0 = const 1 its a dummy feautre to allow us to use a vector formular for the linear regression
#define x0 1.0
#define alpha 0.000001
#define iterations 10000000

void hypothesis(double theta0, double theta1, double x);
double cost(double theta0, double theta1);
void batch_gd(double *theta0, double *theta1);

int main(){
  double theta0 = 0.00;
  double theta1 = 0.00;

  printf("initial cost: %.2f\n", cost(theta0,theta1));

  batch_gd(&theta0, &theta1);

  printf("Final: theta0=%.2f theta1=%.6f\n", theta0, theta1);
  printf("Final cost: %.2f\n", cost(theta0, theta1));

  printf("enter desired sqm in vienna:\n");

  int xi;
  scanf("%d", &xi);
  double x = (double)xi;

  hypothesis(theta0,theta1,x);

  return 0;
}

void hypothesis(double theta0, double theta1, double x){
  double h = (theta1 * x) + (theta0 * 1);
  printf("price %.2f for sqm %.2f\n", h, x);
}

double cost(double theta0, double theta1){
  double jth = 0.0;
  for (int i = 0; i < M; i++){
    jth = jth + pow((theta1 * dataset[i][0]) + (theta0 * x0) - dataset[i][1],2);
  }
  return 0.5 * jth;
}

void batch_gd(double *theta0, double *theta1){
  for (int j = 0; j < iterations; j++){
    double sum0 = 0.0;
    double sum1 = 0.0;

    for (int i = 0; i < M; i++) {
        double x = dataset[i][0];
        double y = dataset[i][1];

        double h = (*theta0) + (*theta1) * x;
        double e = h - y;

        sum0 += e;      // x0 = 1
        sum1 += e * x;  // x1 = x
    }

    *theta0 = *theta0 - alpha * sum0;
    *theta1 = *theta1 - alpha * sum1;
  }
}


