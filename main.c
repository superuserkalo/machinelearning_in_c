#include <stdio.h>
#include <math.h>

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
#define M 60
#define alpha 0.000001
#define iterations 10000000

const double dataset[M][2] = {
    {45.0, 242055.0},
    {45.0, 245790.0},
    {45.0, 275445.0},
    {45.0, 283770.0},
    {45.0, 284535.0},
    {45.0, 285120.0},
    {45.0, 286155.0},
    {45.0, 323370.0},
    {45.0, 372375.0},
    {45.0, 386775.0},
    {45.0, 404190.0},
    {45.0, 408510.0},
    {45.0, 418185.0},
    {45.0, 432180.0},
    {45.0, 473535.0},
    {65.0, 371410.0},
    {65.0, 379210.0},
    {65.0, 410605.0},
    {65.0, 423735.0},
    {65.0, 432575.0},
    {65.0, 440960.0},
    {65.0, 443495.0},
    {65.0, 451685.0},
    {65.0, 456885.0},
    {65.0, 474305.0},
    {65.0, 498875.0},
    {65.0, 511030.0},
    {65.0, 609635.0},
    {65.0, 971945.0},
    {105.0, 459795.0},
    {105.0, 500010.0},
    {105.0, 539175.0},
    {105.0, 570885.0},
    {105.0, 645225.0},
    {105.0, 665175.0},
    {105.0, 686910.0},
    {105.0, 796320.0},
    {105.0, 817425.0},
    {105.0, 841470.0},
    {105.0, 852495.0},
    {105.0, 863940.0},
    {150.0, 661950.0},
    {150.0, 803700.0},
    {150.0, 958650.0},
    {150.0, 967650.0},
    {150.0, 979350.0},
    {150.0, 1136100.0},
    {150.0, 1153200.0},
    {150.0, 1205550.0},
    {150.0, 1449900.0},
    {150.0, 1519500.0},
    {150.0, 1544850.0},
    {150.0, 1558800.0},
    {150.0, 1612500.0},
    {150.0, 1674900.0},
    {150.0, 1887000.0},
    {150.0, 1888350.0},
    {150.0, 2882700.0},
    {150.0, 3411150.0},
    {150.0, 3494400.0}
};

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


