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

#define DATA_X1_NUM 20
#define DATA_X2_NUM 20
#define DATA_NUM DATA_X1_NUM*DATA_X2_NUM
#define M_PI 3.141593
#define GP_WIDTH 0.05
#define INPUT_NUM 2
using namespace Eigen;

double input[INPUT_NUM][DATA_NUM];
double output[DATA_NUM];
double sigma_gp =  0.40;
double beta  =  10.00;
double m     = 0.0;
double sigma = 0.1;
double gaussian_noise(){
	double r1=(double)rand()/RAND_MAX;
	double r2=(double)rand()/RAND_MAX;
    double x_1 = m + sigma*sqrt(-2.0*log(r1))*cos(2*M_PI*r2);
    return  x_1;
}
void create_dataset(void){
	FILE *fp;
	FILE *fp2;
	int i,j;
	double noise; 
	double sinx_cosy_noise[DATA_NUM];
	fp=fopen("sinx_cosy_noise.txt","w");
	fp2=fopen("y_2.txt","w");
	for(i=0;i<DATA_X1_NUM;i++){
		for(j=0;j<DATA_X2_NUM;j++){
			noise = gaussian_noise();
			sinx_cosy_noise[DATA_X2_NUM*i+j]=sin(i*2*M_PI/(DATA_X1_NUM-1))+cos(j*2*M_PI/(DATA_X2_NUM-1))+noise;
			fprintf(fp,"%lf\t%lf\t%lf\n",i*2*M_PI/(DATA_X1_NUM-1),j*2*M_PI/(DATA_X2_NUM-1),sinx_cosy_noise[DATA_X2_NUM*i+j]);
		}
	}
	for(i=0;i<630/4;i++){
		for(j=-200/4;j<200/4;j++){
			fprintf(fp2,"%lf\t%lf\t%lf\n",i*0.01*4,2.0,j*0.01*4);
		}
	}
	fclose(fp);
	fclose(fp2);
}
void read_dataset(void)
{
    FILE *fp;
    double x[INPUT_NUM], y;
    int i=0;
    fp = fopen("sinx_cosy_noise.txt", "r");
    fp = fopen("sinx_cosy_noise.txt", "r");
    if(fp == NULL)exit(0);
    while(fscanf(fp,"%lf%lf%lf", &x[0],&x[1], &y) != EOF ){
		input[0][i]  = x[0];
		input[1][i]  = x[1];
		output[i] = y;
		i++;
    }
    fclose(fp);
}
double kernel(double x1_x, double x1_y , double x2_x, double x2_y){
	return exp(-((x2_x-x1_x)*(x2_x-x1_x)+(x2_y-x1_y)*(x2_y-x1_y))/(2.0*(sigma_gp*sigma_gp)));
}
void GP(){
	FILE *fp_gp,*fp_gp2;
	fp_gp=fopen("gp_20.txt","w");
	fp_gp2=fopen("gp_20of2.txt","w");
    MatrixXd K(DATA_NUM,DATA_NUM);
	MatrixXd C_N(DATA_NUM,DATA_NUM);
	MatrixXd I = MatrixXd::Identity(DATA_NUM,DATA_NUM);
	VectorXd k_star(DATA_NUM);
	VectorXd y(DATA_NUM);
	VectorXd x1(DATA_NUM);
	VectorXd x2(DATA_NUM);

	for(int i=0;i<DATA_NUM;i++){
		x1(i) = input[0][i];
		x2(i) = input[1][i];
    }
	for(int i=0;i<DATA_NUM;i++){
        y(i) = output[i];
    }
	for(int i=0;i<DATA_NUM;i++){
        for(int j=0;j<DATA_NUM;j++) K(i,j)=kernel(x1(i),x2(i),x1(j),x2(j));
    }
	C_N = K + I/beta;
	FullPivLU < MatrixXd > lu(C_N);
	MatrixXd C_N_inv=lu.inverse();
	VectorXd m((int)((2*M_PI/GP_WIDTH+1)*(2*M_PI/GP_WIDTH)+1));
	VectorXd v((int)((2*M_PI/GP_WIDTH+1)*(2*M_PI/GP_WIDTH+1)));
	for(double i=0;i<2*M_PI;i+=GP_WIDTH){
		for(double i2=0;i2<2*M_PI;i2+=GP_WIDTH){
			for(int j=0;j<DATA_NUM;j++){
				k_star(j)=kernel(i,i2,x1[j],x2[j]);
			}
			m((int)((2*M_PI/GP_WIDTH)*(i/GP_WIDTH))+(i2/GP_WIDTH)) =  k_star.transpose() * C_N_inv * y;
			v((int)((2*M_PI/GP_WIDTH)*(i/GP_WIDTH))+(i2/GP_WIDTH)) =  kernel(i,i2,i,i2) - k_star.transpose() * C_N_inv * k_star;
		}
	}
	for(double i=0;i<2*M_PI;i+=GP_WIDTH)for(double i2=0;i2<2*M_PI;i2+=GP_WIDTH)fprintf(fp_gp,"%lf\t%lf\t%lf\t%lf\t%lf\n",i,i2,m((int)((2*M_PI/GP_WIDTH)*(i/GP_WIDTH))+(i2/GP_WIDTH)),m((int)((2*M_PI/GP_WIDTH)*(i/GP_WIDTH))+(i2/GP_WIDTH))+pow(v((int)((2*M_PI/GP_WIDTH)*(i/GP_WIDTH))+(i2/GP_WIDTH)),0.5),m((int)((2*M_PI/GP_WIDTH)*(i/GP_WIDTH))+(i2/GP_WIDTH))-pow(v((int)((2*M_PI/GP_WIDTH)*(i/GP_WIDTH))+(i2/GP_WIDTH)),0.5));
	//’f–Ê‰æ‘œì¬—p
	for(double i=0;i<2*M_PI;i+=GP_WIDTH){
		double i3=2.0;//y=2.0‚Ì’f–Ê
		fprintf(fp_gp2,"%lf\t%lf\t%lf\t%lf\t%lf\n",i,i3,m((int)((2*M_PI/GP_WIDTH)*(i/GP_WIDTH))+(i3/GP_WIDTH)),m((int)((2*M_PI/GP_WIDTH)*(i/GP_WIDTH))+(i3/GP_WIDTH))+pow(v((int)((2*M_PI/GP_WIDTH)*(i/GP_WIDTH))+(i3/GP_WIDTH)),0.5),m((int)((2*M_PI/GP_WIDTH)*(i/GP_WIDTH))+(i3/GP_WIDTH))-pow(v((int)((2*M_PI/GP_WIDTH)*(i/GP_WIDTH))+(i3/GP_WIDTH)),0.5));
	}
	fclose(fp_gp);
	fclose(fp_gp2);
}
void main(){
	srand((unsigned)time(NULL));
	create_dataset();
	read_dataset();
	GP();
}	



