#pragma once
#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>
#include<random>
#include <cstdlib>

using namespace std;
using namespace cv;



//重构误差,R为秩一张量的个数
double errorNum(vector<Mat> chi, int I, int J, int K, int R, Mat U, Mat V, Mat W, double a3) {
    double p1 = 0, p2 = 0, p3 = 0, p4 = 0;
    double sub = 0;

    //imshow("123", chi[0]);
    ////waitKey(0);

    //chi[0].copyTo(U);
    ////imshow("456", U);
    ////waitKey(0);

    //cout << chi[0].type()<<endl;
    //cout << chi[0];



    //第一部分  差
    for (int i = 0; i < I; i++) {
        for (int j = 0; j < J; j++) {
            for (int k = 0; k < K; k++) {
                //对第ijk个元素操作

                double temp = 0, add = 0;
                //之一张量之和
                for (int r = 0; r < R; r++) {
                    add += U.at<double>(i, r) * V.at<double>(j, r) * W.at<double>(k, r);
                }

                //U.at<double>(i, 0);
                //chi[k].at<double>(i, j);

                temp = chi[k].at<double>(i, j) - add;
                p1 += temp * temp;//第ijk上元素的平方
                temp = 0;
                add = 0;
            }
        }
    }

    //第二、三部分U,V
    Mat Uturn = U.t() * U;//均为R*R的矩阵
    Mat Vturn = V.t() * V;

    for (int i = 0; i < R; i++) {
        for (int j = 0; j < R; j++) {
            p2 += Uturn.at<double>(i, j);
            p3 += Vturn.at<double>(i, j);
        }
    }

    //第四部分W
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < R; j++) {
            p4 += W.at<double>(i, j) * W.at<double>(i, j);
        }
    }

    //总和
    sub = 0.5 * p1 + 0.5 * p2 + 0.5 * p3 + a3 * p4;
    return sub;

}



//CP_decomposition(本文)
void CP_decomposition_own(vector<Mat>& chi, int n, int R, Mat& U, Mat& V, Mat& W, double yibu = 10, int maxiter = 50, double a3 = 0.05,float beta=0.9) {
    //初始化
    Mat Ut, Vt, Wt, Uup, Vup, Wup, Udown, Vdown, Wdown;


    int I = chi[0].rows;
    int J = chi[0].cols;
    int K = n;
    int k = 0;//迭代次数
    U.create(I, R, CV_64F);
    V.create(J, R, CV_64F);
    W.create(K, R, CV_64F);
    Uup.create(I, R, CV_64F);//存储上部分之和
    Vup.create(J, R, CV_64F);
    Wup.create(K, R, CV_64F);
    Udown.create(I, R, CV_64F);//存储下部分之和
    Vdown.create(J, R, CV_64F);
    Wdown.create(K, R, CV_64F);
    randu(U, Scalar::all(0), Scalar::all(1));
    randu(V, Scalar::all(0), Scalar::all(1));
    randu(W, Scalar::all(0), Scalar::all(1));
    randu(Uup, Scalar::all(0), Scalar::all(0));
    randu(Vup, Scalar::all(0), Scalar::all(0));
    randu(Wup, Scalar::all(0), Scalar::all(0));
    randu(Udown, Scalar::all(0), Scalar::all(0));
    randu(Vdown, Scalar::all(0), Scalar::all(0));
    randu(Wdown, Scalar::all(0), Scalar::all(0));
    Ut = U.clone();//存储迭代后的数据
    Vt = V.clone();
    Wt = W.clone();



    //开始迭代
    while (errorNum(chi, I, J, K, R, U, V, W, 0.005) > yibu && k < maxiter) {
        //对U进行迭代存放进Ut
        for (int i = 0; i < I; i++) {
            for (int j = 0; j < R; j++) {

                double top = 0, bottom = 0;
                //上部分
                for (int s = 0; s < J; s++) {
                    for (int t = 0; t < K; t++) {
                        top += chi[t].at<double>(i, s) * V.at<double>(s, j) * W.at<double>(t, j);

                    }
                }

                //下部分
                for (int m = 0; m < R; m++) {
                    bottom += U.at<double>(i, m) * (V.col(m).dot(V.col(j))) * (W.col(m).dot(W.col(j))) + U.at<double>(i, m);

                }

                //迭代(先给上下部分和分别乘0.9，再加上这次的)
                Uup.at<double>(i, j) *= beta;
                Udown.at<double>(i, j) *= beta;
                Uup.at<double>(i, j) += top;
                Udown.at<double>(i, j) += bottom;




                Ut.at<double>(i, j) = U.at<double>(i, j) * Uup.at<double>(i, j) / Udown.at<double>(i, j);


            }
        }






        //对V进行迭代
        for (int s = 0; s < J; s++) {
            for (int j = 0; j < R; j++) {

                double top = 0, bottom = 0;
                //上部分
                for (int i = 0; i < I; i++) {
                    for (int t = 0; t < K; t++) {
                        top += chi[t].at<double>(i, s) * U.at<double>(i, j) * W.at<double>(t, j);

                    }
                }

                //下部分
                for (int m = 0; m < R; m++) {
                    bottom += V.at<double>(s, m) * (U.col(m).dot(U.col(j))) * (W.col(m).dot(W.col(j))) + V.at<double>(s, m);

                }


                //迭代(先给上下部分和分别乘0.9，再加上这次的)
                Vup.at<double>(s, j) *= beta;
                Vdown.at<double>(s, j) *= beta;
                Vup.at<double>(s, j) += top;
                Vdown.at<double>(s, j) += bottom;

                Vt.at<double>(s, j) = V.at<double>(s, j) * Vup.at<double>(s, j) / Vdown.at<double>(s, j);



            }
        }



        //对W进行迭代
        for (int t = 0; t < K; t++) {
            for (int j = 0; j < R; j++) {

                double top = 0, bottom = 0;
                //上部分
                for (int i = 0; i < I; i++) {
                    for (int s = 0; s < J; s++) {
                        top += chi[t].at<double>(i, s) * U.at<double>(i, j) * V.at<double>(s, j);

                    }
                }

                //下部分
                for (int m = 0; m < R; m++) {
                    bottom += W.at<double>(t, m) * (U.col(m).dot(U.col(j))) * (V.col(m).dot(V.col(j)));

                }
                bottom += 2 * a3 * W.at<double>(t, j);//加上后半部分



                //迭代(先给上下部分和分别乘0.9，再加上这次的)
                Wup.at<double>(t, j) *= beta;
                Wdown.at<double>(t, j) *= beta;
                Wup.at<double>(t, j) += top;
                Wdown.at<double>(t, j) += bottom;




                Wt.at<double>(t, j) = W.at<double>(t, j) * Wup.at<double>(t, j)/ Wdown.at<double>(t, j);

                if (j == 10)
                    cout << Wt.at<double>(t, j) << endl;


                if (k == 11)
                    waitKey(0);


            }




        }





        //更新U矩阵的数据
        U = Ut.clone();
        //更新V矩阵的数据
        V = Vt.clone();
        //更新V矩阵的数据
        W = Wt.clone();
        //迭代器加一
        k++;



        //if (!(k % 10))
        cout << "第" << k << "次迭代后重构误差：" << errorNum(chi, I, J, K, R, U, V, W, 0.005) << endl;

    }

    cout << Wt << endl;

    cout << "分解完成" << endl;




}
