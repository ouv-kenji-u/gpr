/******************************************************************************
 * @author     Kenji Urai <keng.ouv.cv@gmail.com>
 * @copyright  2018 Kenji Urai
 * @license    http://www.opensource.org/licenses/mit-license.html  MIT License
*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "Eigen/Core"
#include "Eigen/LU"

#define SUM_NUM 1000
#define DATA_NUM 10
#define M_PI 3.141592653589793238
#define GP_WIDTH 0.02
using namespace Eigen;

double input[DATA_NUM];
double output[DATA_NUM];
double sigma_gp =  0.40;
double beta  =  10.00;

double m=0.;
double sigma=0.1;

double gaussian_noise(){
	double r1=(double)rand()/RAND_MAX;
	double r2=(double)rand()/RAND_MAX;
    double x_1 = m + sigma*sqrt(-2.0*log(r1))*cos(2*M_PI*r2);
    return  x_1;
}

void create_dataset(void){
	FILE *fp;
	int i;
	double noise; 
	double sin_noise[DATA_NUM];
	fp=fopen("sin_noise_10.txt","w");
	for(i=0;i<DATA_NUM;i++){
		noise = gaussian_noise();
		sin_noise[i]=sin(i*2*M_PI/(DATA_NUM-1))+noise;
	}
	for(i=0;i<DATA_NUM;i++){
		fprintf(fp,"%lf\t%lf\n",i*2*M_PI/(DATA_NUM-1),sin_noise[i]);
	}
	fclose(fp);
}
void read_dataset(void)
{
    FILE *fp;
    double x, y;
    int i=0;
    fp = fopen("sin_noise_10.txt", "r");
    if(fp == NULL)exit(0);
    while(fscanf(fp,"%lf%lf", &x, &y) != EOF ){
		input[i]  = x;
		output[i] = y;
		i++;
    }
    fclose(fp);
}
double kernel(double x1, double x2){
	return exp(-((x1-x2)*(x1-x2))/(2.0*(sigma_gp*sigma_gp)));
}
void GP(){
	FILE *fp_gp;
	fp_gp=fopen("gp_1_10.txt","w");
    MatrixXd K(DATA_NUM,DATA_NUM);
	MatrixXd C_N(DATA_NUM,DATA_NUM);
	MatrixXd I = MatrixXd::Identity(DATA_NUM,DATA_NUM);
	VectorXd k_star(DATA_NUM);
	VectorXd y(DATA_NUM);
	for(int i=0;i<DATA_NUM;i++){
        y(i)=output[i];
    }
	for(int i=0;i<DATA_NUM;i++){
        for(int j=0;j<DATA_NUM;j++) K(i,j)=kernel(input[i],input[j]);
    }
	C_N = K + I/beta;
	FullPivLU< MatrixXd > lu(C_N);
    MatrixXd C_N_inv=lu.inverse();

	VectorXd m((int)(2*M_PI/GP_WIDTH));
	for(double i=0;i<2*M_PI;i+=GP_WIDTH){
		for(int j=0;j<DATA_NUM;j++){
			k_star(j)=kernel(i,input[j]);
		}
		m((int)(i/GP_WIDTH)) =  k_star.transpose() * C_N_inv * y;
	}
	VectorXd v((int)(2*M_PI/GP_WIDTH));
	for(double i=0;i<2*M_PI;i+=GP_WIDTH){
		for(int j=0;j<DATA_NUM;j++){
			k_star(j)=kernel(i,input[j]);
		}
		v((int)(i/GP_WIDTH)) =  kernel(i,i) - k_star.transpose() * C_N_inv * k_star;
	}
	for(double i=0;i<2*M_PI;i+=GP_WIDTH)fprintf(fp_gp,"%lf\t%lf\t%lf\t%lf\n",i,m((int)(i/GP_WIDTH)),m((int)(i/GP_WIDTH))+pow(v((int)(i/GP_WIDTH)),0.5),m((int)(i/GP_WIDTH))-pow(v((int)(i/GP_WIDTH)),0.5));
	fclose(fp_gp);
}
void main(){
	srand((unsigned)time(NULL));
	create_dataset();
	read_dataset();
	GP();
}