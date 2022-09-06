#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>
#include<random>
#include <cstdlib>

#define _WINSOCK_DEPRECATED_NO_WARNINGS

using namespace std;
using namespace cv;

//LBP提取
template <typename _tp>
void getOriginLBPFeature(InputArray _src, OutputArray _dst)
{
    Mat src = _src.getMat();
    _dst.create(src.rows - 2, src.cols - 2, CV_8UC1);
    Mat dst = _dst.getMat();
    dst.setTo(0);
    for (int i = 1; i < src.rows - 1; i++)
    {
        for (int j = 1; j < src.cols - 1; j++)
        {
            _tp center = src.at<_tp>(i, j);
            unsigned char lbpCode = 0;
            lbpCode |= (src.at<_tp>(i - 1, j - 1) > center) << 7;
            lbpCode |= (src.at<_tp>(i - 1, j) > center) << 6;
            lbpCode |= (src.at<_tp>(i - 1, j + 1) > center) << 5;
            lbpCode |= (src.at<_tp>(i, j + 1) > center) << 4;
            lbpCode |= (src.at<_tp>(i + 1, j + 1) > center) << 3;
            lbpCode |= (src.at<_tp>(i + 1, j) > center) << 2;
            lbpCode |= (src.at<_tp>(i + 1, j - 1) > center) << 1;
            lbpCode |= (src.at<_tp>(i, j - 1) > center) << 0;
            dst.at<uchar>(i - 1, j - 1) = lbpCode;
        }
    }
}


//MB-LBP提取，scale为小区域的大小*3
void getMultiScaleBlockLBPFeature(InputArray _src, OutputArray _dst, int scale=15)
{
    Mat src = _src.getMat();
    Mat dst = _dst.getMat();
    //定义并计算积分图像
    int cellSize = scale / 3;
    int offset = cellSize / 2;
    Mat cellImage(src.rows - 2 * offset, src.cols - 2 * offset, CV_8UC1);
    for (int i = offset; i < src.rows - offset; i++)
    {
        for (int j = offset; j < src.cols - offset; j++)
        {
            int temp = 0;
            for (int m = -offset; m < offset + 1; m++)
            {
                for (int n = -offset; n < offset + 1; n++)
                {
                    temp += src.at<uchar>(i + n, j + m);
                }
            }
            temp /= (cellSize * cellSize);
            cellImage.at<uchar>(i - cellSize / 2, j - cellSize / 2) = uchar(temp);
        }
    }
    getOriginLBPFeature<uchar>(cellImage, _dst);
}

//读取文件夹下的所有文件
std::vector<cv::Mat> ReadImage(cv::String pattern)
{
    std::vector<cv::String> fn;
    cv::glob(pattern, fn, false);
    std::vector<cv::Mat> images;
    int count = fn.size(); //number of png files in images folder
    for (int i = 0; i < count; i++)
    {
        images.emplace_back(cv::imread(fn[i]));
    }
    return images;
}

//将orl_chlight文件夹下的400张图片依次读出提取MB-LBP特征，并写入orl_MBLBP文件夹下(LBP主程序)
void extractLBP() {
    String inurl, outurl = "";
    Mat  src, src_gray, LBP_src, MB_LBPsrc; //定义图片
    for (int i = 1; i < 41; i++) {
        for (int j = 0; j < 10; j++) {
            inurl = "./orl_chlight/s" + to_string(i) + '_' + to_string(j) + ".jpg";
            src = imread(inurl);//读入图片
            if (src.empty())  //判断图片是否为空
            {
                cout << "could not load image1" <<endl;
                //return -1;
            }
            cout << inurl << endl;
            cvtColor(src, src_gray, COLOR_BGR2GRAY);//转换为灰度图
            getMultiScaleBlockLBPFeature(src_gray, MB_LBPsrc, 51);

            outurl = "./orl_MBLBP17/s" + to_string(i) + '_' + to_string(j) + ".jpg";

            imwrite(outurl, MB_LBPsrc);

        }
    }
}





//读取n张图片便成一个张量,从下标begin开始取n个
void img2chi(vector<Mat> &chi, int file, int begin, int n) {
    String inurl="";

    for (int j = 0; j < n; j++) {        
        inurl = ".\\orl_cut\\s"+to_string(file)+"_"+ to_string(j+begin) + ".jpg";
        chi[j] = imread(inurl);//读入图片
        if (chi[j].empty())  //判断图片是否为空
        {
            cout << "could not load image1" << endl;
        }
        cout << inurl <<"已被取出" << endl;


        
    }

}



//重构误差,R为秩一张量的个数
double errorNum(vector<Mat> chi, int I,int J,int K, int R, Mat U, Mat V, Mat W,double a3) {
    double p1=0,p2=0,p3=0,p4= 0;
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

                double temp=0, add = 0;
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



//CP_decomposition(宋珊文章:乘性迭代规则)
void CP_decomposition(vector<Mat> &chi,int n,int R,Mat &U,Mat &V, Mat &W, double yibu=10,int maxiter=100,double a3=0.05) {
    //初始化
    Mat Ut,Vt,Wt;

    int I = chi[0].rows;
    int J = chi[0].cols;
    int K = n;
    int k = 0;//迭代次数
    U.create(I, R, CV_64FC1);
    V.create(J, R, CV_64FC1);
    W.create(K, R, CV_64FC1);
    randu(U, Scalar::all(0), Scalar::all(1));
    randu(V, Scalar::all(0), Scalar::all(1));
    randu(W, Scalar::all(0), Scalar::all(1));
    Ut = U.clone();//存储迭代后的数据
    Vt = V.clone();
    Wt = W.clone();


    
    double errornumpre = errorNum(chi, I, J, K, R, U, V, W, a3);
    //开始迭代
    while (errorNum(chi, I, J, K, R, U, V, W, a3) > yibu && k < maxiter) {
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

                //迭代

                
                Ut.at<double>(i, j) = U.at<double>(i, j)*top/bottom;


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

                //迭代

                Vt.at<double>(s, j) = V.at<double>(s, j) * top / bottom;

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

                //迭代
                Wt.at<double>(t, j) = W.at<double>(t, j) * (top / (bottom + 2 * a3 * W.at<double>(t, j)));

                if(j==10)
                    cout << Wt.at<double>(t, j) << endl;

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
        
        
        cout << "第" << k << "次迭代后重构误差：" << errorNum(chi, I, J, K, R, U, V, W, a3) << endl;

        //当不能下降时停止迭代（可选择）
        double errornum = errorNum(chi, I, J, K, R, U, V, W, a3);
        if ((errornumpre - errornum) < 0.001)break;
        errornumpre = errornum;

            

    }

            cout << Wt << endl;

    cout << "分解完成" << endl;




}

//CP_decomposition(本文)
void CP_decomposition_own(vector<Mat>& chi, int n, int R, Mat& U, Mat& V, Mat& W, double yibu = 1000, int maxiter = 50, double a3 = 0.05, float beta = 0.9) {
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

    double errornumpre = errorNum(chi, I, J, K, R, U, V, W, 0.005);

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




                Wt.at<double>(t, j) = W.at<double>(t, j) * Wup.at<double>(t, j) / Wdown.at<double>(t, j);

                if (j == 10)
                    cout << Wt.at<double>(t, j) << endl;////////////////////////////////////////
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


        //当不能下降时停止迭代（可选择）
        double errornum = errorNum(chi, I, J, K, R, U, V, W, a3);
        if ((errornumpre - errornum) < 0.001)break;
        errornumpre = errornum;

        //if (!(k % 10))
        cout << "第" << k << "次迭代后重构误差：" << errorNum(chi, I, J, K, R, U, V, W, a3) << endl;

    }


    cout << "分解完成" << endl;

}




//低秩表示
Mat dizhi(vector<Mat> chi,int n,int R,Mat U,Mat V,Mat W) {
    Mat C;
    int I = chi[0].rows;
    int J = chi[0].cols;
    int K = n;

    C.create(I*J, R, CV_64F);

    //Khatri-Rao积  C
    for (int r = 0; r < R; r++) {
        for (int j = 0; j < J; j++) {
            for (int i = 0; i < I; i++) {
                C.at<double>(j * J + i, r) = U.at<double>(i, r) * V.at<double>(j, r);

            }
        }
    }

    Mat P, Q;
    P.create(I * J, K, CV_64F);
    Q.create(R, K,CV_64F);

    P = C * W.t();
    Q = (C.t() * C).inv() * C.t() * P;


    cout << "低秩表示完成" << endl;
    return Q;


}


//余弦相似性
float cos_simple(Mat Q1, Mat Q2) {

    double up = 0,down1=0,down2=0;

    up = Q1.dot(Q2);
    //for (int i = 0; i < Q1.rows;i++) {
    //    for (int j = 0; j < Q1.cols; j++) {
    //        up += Q1.at<double>(i, j) * Q2.at<double>(i, j);
    //    }
    //}

    for (int i = 0; i < Q1.rows;i++) {
        for (int j = 0; j < Q1.cols; j++) {
            down1 += Q1.at<double>(i, j) * Q1.at<double>(i, j);
            down2 += Q2.at<double>(i, j) * Q2.at<double>(i, j);
        }
    }

    return up / (sqrt(down1) * sqrt(down2));



}




int main()
{
    vector<Mat> chi_1(4), chi_2(4),im1(4),im2(4);
    //Mat chi_1[4], chi_2[4] ,src;
    int R = 30;
    int n = 4;
    int file = 1;
    int success = 0;//成功个数

    for (int i = 0; i < 4; i++) {
        chi_1[i].create(112, 92, CV_64FC1);
        chi_2[i].create(112, 92, CV_64FC1);
    }


    for (int i = 1; i < 41; i++) {
        img2chi(chi_1, i, 5, n);
        img2chi(chi_2, i, 1, n);

        for (int i = 0; i < 4; i++) {
            chi_1[i].convertTo(im1[i], CV_64FC1, 1.0 / 255.0);
            chi_2[i].convertTo(im2[i], CV_64FC1, 1.0 / 255.0);
        }


        Mat U1, V1, W1, U2, V2, W2;

        cout << "chi_1正在分解中:" << endl;
        CP_decomposition_own(im1, n, R, U1, V1, W1);
        cout << "chi_2正在分解中:" << endl;
        CP_decomposition_own(im2, n, R, U2, V2, W2);

        Mat Q1 = dizhi(chi_1, n, R, U1, V1, W1);
        Mat Q2 = dizhi(chi_2, n, R, U2, V2, W2);

        float yuxian = cos_simple(Q1, Q2);

        cout << "第" + to_string(i) + "个文件夹中余弦相似性为：" << yuxian;

        if (yuxian > 0.7)
            success++;
    }

    cout << "识别率：" << success / 40<<endl;
















    //Mat src;
    //    src=imread("./image_change/s26/1.jpg");//读入图片
    //Mat src_gray, LBP_src, MB_LBPsrc;
    //if (src.empty())  //判断图片是否为空
    //{
    //    cout << "could not load image1" << endl;
    //    return -1;
    //}

    //imshow("123", src);

    //cvtColor(src, src_gray, COLOR_BGR2GRAY);//转换为灰度图
    //imshow("output", src_gray);//输出灰度图

    //getOriginLBPFeature<uchar>(src_gray, LBP_src);
    //imshow("LBP", LBP_src);

    //getMultiScaleBlockLBPFeature(src_gray, MB_LBPsrc, 9);
    //imshow("MB_LBP_9", MB_LBPsrc);

    //getMultiScaleBlockLBPFeature(src_gray, MB_LBPsrc, 15);
    //imshow("MB_LBP_15", MB_LBPsrc);

    //getMultiScaleBlockLBPFeature(src_gray, MB_LBPsrc, 21);
    //imshow("MB_LBP_21", MB_LBPsrc);


    //waitKey(0); //可以让输出窗口一直显示
    return 0;
}



