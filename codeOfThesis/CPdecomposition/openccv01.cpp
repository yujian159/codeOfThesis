#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>
#include<random>
#include <cstdlib>

#define _WINSOCK_DEPRECATED_NO_WARNINGS

using namespace std;
using namespace cv;

//LBP��ȡ
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


//MB-LBP��ȡ��scaleΪС����Ĵ�С*3
void getMultiScaleBlockLBPFeature(InputArray _src, OutputArray _dst, int scale=15)
{
    Mat src = _src.getMat();
    Mat dst = _dst.getMat();
    //���岢�������ͼ��
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

//��ȡ�ļ����µ������ļ�
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

//��orl_chlight�ļ����µ�400��ͼƬ���ζ�����ȡMB-LBP��������д��orl_MBLBP�ļ�����(LBP������)
void extractLBP() {
    String inurl, outurl = "";
    Mat  src, src_gray, LBP_src, MB_LBPsrc; //����ͼƬ
    for (int i = 1; i < 41; i++) {
        for (int j = 0; j < 10; j++) {
            inurl = "./orl_chlight/s" + to_string(i) + '_' + to_string(j) + ".jpg";
            src = imread(inurl);//����ͼƬ
            if (src.empty())  //�ж�ͼƬ�Ƿ�Ϊ��
            {
                cout << "could not load image1" <<endl;
                //return -1;
            }
            cout << inurl << endl;
            cvtColor(src, src_gray, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
            getMultiScaleBlockLBPFeature(src_gray, MB_LBPsrc, 51);

            outurl = "./orl_MBLBP17/s" + to_string(i) + '_' + to_string(j) + ".jpg";

            imwrite(outurl, MB_LBPsrc);

        }
    }
}





//��ȡn��ͼƬ���һ������,���±�begin��ʼȡn��
void img2chi(vector<Mat> &chi, int file, int begin, int n) {
    String inurl="";

    for (int j = 0; j < n; j++) {        
        inurl = ".\\orl_cut\\s"+to_string(file)+"_"+ to_string(j+begin) + ".jpg";
        chi[j] = imread(inurl);//����ͼƬ
        if (chi[j].empty())  //�ж�ͼƬ�Ƿ�Ϊ��
        {
            cout << "could not load image1" << endl;
        }
        cout << inurl <<"�ѱ�ȡ��" << endl;


        
    }

}



//�ع����,RΪ��һ�����ĸ���
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



    //��һ����  ��
    for (int i = 0; i < I; i++) {
        for (int j = 0; j < J; j++) {
            for (int k = 0; k < K; k++) {
                //�Ե�ijk��Ԫ�ز���

                double temp=0, add = 0;
                //֮һ����֮��
                for (int r = 0; r < R; r++) {
                    add += U.at<double>(i, r) * V.at<double>(j, r) * W.at<double>(k, r);
                }

                //U.at<double>(i, 0);
                //chi[k].at<double>(i, j);

                temp = chi[k].at<double>(i, j) - add;
                p1 += temp * temp;//��ijk��Ԫ�ص�ƽ��
                temp = 0;
                add = 0;
            }
        }
    }

    //�ڶ���������U,V
    Mat Uturn = U.t() * U;//��ΪR*R�ľ���
    Mat Vturn = V.t() * V;

    for (int i = 0; i < R; i++) {
        for (int j = 0; j < R; j++) {
            p2 += Uturn.at<double>(i, j);
            p3 += Vturn.at<double>(i, j);
        }
    }

    //���Ĳ���W
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < R; j++) {
            p4 += W.at<double>(i, j) * W.at<double>(i, j);
        }
    }

    //�ܺ�
    sub = 0.5 * p1 + 0.5 * p2 + 0.5 * p3 + a3 * p4;
    return sub;

}



//CP_decomposition(��ɺ����:���Ե�������)
void CP_decomposition(vector<Mat> &chi,int n,int R,Mat &U,Mat &V, Mat &W, double yibu=10,int maxiter=100,double a3=0.05) {
    //��ʼ��
    Mat Ut,Vt,Wt;

    int I = chi[0].rows;
    int J = chi[0].cols;
    int K = n;
    int k = 0;//��������
    U.create(I, R, CV_64FC1);
    V.create(J, R, CV_64FC1);
    W.create(K, R, CV_64FC1);
    randu(U, Scalar::all(0), Scalar::all(1));
    randu(V, Scalar::all(0), Scalar::all(1));
    randu(W, Scalar::all(0), Scalar::all(1));
    Ut = U.clone();//�洢�����������
    Vt = V.clone();
    Wt = W.clone();


    
    double errornumpre = errorNum(chi, I, J, K, R, U, V, W, a3);
    //��ʼ����
    while (errorNum(chi, I, J, K, R, U, V, W, a3) > yibu && k < maxiter) {
                //��U���е�����Ž�Ut
        for (int i = 0; i < I; i++) {
            for (int j = 0; j < R; j++) {

                double top = 0, bottom = 0;
                //�ϲ���
                for (int s = 0; s < J; s++) {
                    for (int t = 0; t < K; t++) {
                        top += chi[t].at<double>(i, s) * V.at<double>(s, j) * W.at<double>(t, j);

                    }
                }

                //�²���
                for (int m = 0; m < R; m++) {
                    bottom += U.at<double>(i, m) * (V.col(m).dot(V.col(j))) * (W.col(m).dot(W.col(j))) + U.at<double>(i, m);

                }

                //����

                
                Ut.at<double>(i, j) = U.at<double>(i, j)*top/bottom;


            }
        }


        //��V���е���
        for (int s = 0; s < J; s++) {
            for (int j = 0; j < R; j++) {

                double top = 0, bottom = 0;
                //�ϲ���
                for (int i = 0; i < I; i++) {
                    for (int t = 0; t < K; t++) {
                        top += chi[t].at<double>(i, s) * U.at<double>(i, j) * W.at<double>(t, j);

                    }
                }

                //�²���
                for (int m = 0; m < R; m++) {
                    bottom += V.at<double>(s, m) * (U.col(m).dot(U.col(j))) * (W.col(m).dot(W.col(j))) + V.at<double>(s, m);

                }

                //����

                Vt.at<double>(s, j) = V.at<double>(s, j) * top / bottom;

            }
        }



        //��W���е���
        for (int t = 0; t < K; t++) {
            for (int j = 0; j < R; j++) {

                double top = 0, bottom = 0;
                //�ϲ���
                for (int i = 0; i < I; i++) {
                    for (int s = 0; s < J; s++) {
                        top += chi[t].at<double>(i, s) * U.at<double>(i, j) * V.at<double>(s, j);

                    }
                }

                //�²���
                for (int m = 0; m < R; m++) {
                    bottom += W.at<double>(t, m) * (U.col(m).dot(U.col(j))) * (V.col(m).dot(V.col(j)));

                }

                //����
                Wt.at<double>(t, j) = W.at<double>(t, j) * (top / (bottom + 2 * a3 * W.at<double>(t, j)));

                if(j==10)
                    cout << Wt.at<double>(t, j) << endl;

            }


        }


        


        //����U���������
        U = Ut.clone();
        //����V���������
        V = Vt.clone();
        //����V���������
        W = Wt.clone();
        //��������һ
        k++;
        
        
        cout << "��" << k << "�ε������ع���" << errorNum(chi, I, J, K, R, U, V, W, a3) << endl;

        //�������½�ʱֹͣ��������ѡ��
        double errornum = errorNum(chi, I, J, K, R, U, V, W, a3);
        if ((errornumpre - errornum) < 0.001)break;
        errornumpre = errornum;

            

    }

            cout << Wt << endl;

    cout << "�ֽ����" << endl;




}

//CP_decomposition(����)
void CP_decomposition_own(vector<Mat>& chi, int n, int R, Mat& U, Mat& V, Mat& W, double yibu = 1000, int maxiter = 50, double a3 = 0.05, float beta = 0.9) {
    //��ʼ��
    Mat Ut, Vt, Wt, Uup, Vup, Wup, Udown, Vdown, Wdown;


    int I = chi[0].rows;
    int J = chi[0].cols;
    int K = n;
    int k = 0;//��������
    U.create(I, R, CV_64F);
    V.create(J, R, CV_64F);
    W.create(K, R, CV_64F);
    Uup.create(I, R, CV_64F);//�洢�ϲ���֮��
    Vup.create(J, R, CV_64F);
    Wup.create(K, R, CV_64F);
    Udown.create(I, R, CV_64F);//�洢�²���֮��
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
    Ut = U.clone();//�洢�����������
    Vt = V.clone();
    Wt = W.clone();

    double errornumpre = errorNum(chi, I, J, K, R, U, V, W, 0.005);

    //��ʼ����
    while (errorNum(chi, I, J, K, R, U, V, W, 0.005) > yibu && k < maxiter) {
        //��U���е�����Ž�Ut
        for (int i = 0; i < I; i++) {
            for (int j = 0; j < R; j++) {

                double top = 0, bottom = 0;
                //�ϲ���
                for (int s = 0; s < J; s++) {
                    for (int t = 0; t < K; t++) {
                        top += chi[t].at<double>(i, s) * V.at<double>(s, j) * W.at<double>(t, j);

                    }
                }

                //�²���
                for (int m = 0; m < R; m++) {
                    bottom += U.at<double>(i, m) * (V.col(m).dot(V.col(j))) * (W.col(m).dot(W.col(j))) + U.at<double>(i, m);

                }

                //����(�ȸ����²��ֺͷֱ��0.9���ټ�����ε�)
                Uup.at<double>(i, j) *= beta;
                Udown.at<double>(i, j) *= beta;
                Uup.at<double>(i, j) += top;
                Udown.at<double>(i, j) += bottom;


                Ut.at<double>(i, j) = U.at<double>(i, j) * Uup.at<double>(i, j) / Udown.at<double>(i, j);


            }
        }


        //��V���е���
        for (int s = 0; s < J; s++) {
            for (int j = 0; j < R; j++) {

                double top = 0, bottom = 0;
                //�ϲ���
                for (int i = 0; i < I; i++) {
                    for (int t = 0; t < K; t++) {
                        top += chi[t].at<double>(i, s) * U.at<double>(i, j) * W.at<double>(t, j);

                    }
                }

                //�²���
                for (int m = 0; m < R; m++) {
                    bottom += V.at<double>(s, m) * (U.col(m).dot(U.col(j))) * (W.col(m).dot(W.col(j))) + V.at<double>(s, m);

                }


                //����(�ȸ����²��ֺͷֱ��0.9���ټ�����ε�)
                Vup.at<double>(s, j) *= beta;
                Vdown.at<double>(s, j) *= beta;
                Vup.at<double>(s, j) += top;
                Vdown.at<double>(s, j) += bottom;

                Vt.at<double>(s, j) = V.at<double>(s, j) * Vup.at<double>(s, j) / Vdown.at<double>(s, j);



            }
        }



        //��W���е���
        for (int t = 0; t < K; t++) {
            for (int j = 0; j < R; j++) {

                double top = 0, bottom = 0;
                //�ϲ���
                for (int i = 0; i < I; i++) {
                    for (int s = 0; s < J; s++) {
                        top += chi[t].at<double>(i, s) * U.at<double>(i, j) * V.at<double>(s, j);

                    }
                }

                //�²���
                for (int m = 0; m < R; m++) {
                    bottom += W.at<double>(t, m) * (U.col(m).dot(U.col(j))) * (V.col(m).dot(V.col(j)));

                }
                bottom += 2 * a3 * W.at<double>(t, j);//���Ϻ�벿��



                //����(�ȸ����²��ֺͷֱ��0.9���ټ�����ε�)
                Wup.at<double>(t, j) *= beta;
                Wdown.at<double>(t, j) *= beta;
                Wup.at<double>(t, j) += top;
                Wdown.at<double>(t, j) += bottom;




                Wt.at<double>(t, j) = W.at<double>(t, j) * Wup.at<double>(t, j) / Wdown.at<double>(t, j);

                if (j == 10)
                    cout << Wt.at<double>(t, j) << endl;////////////////////////////////////////
            }

        }

        //����U���������
        U = Ut.clone();
        //����V���������
        V = Vt.clone();
        //����V���������
        W = Wt.clone();
        //��������һ
        k++;


        //�������½�ʱֹͣ��������ѡ��
        double errornum = errorNum(chi, I, J, K, R, U, V, W, a3);
        if ((errornumpre - errornum) < 0.001)break;
        errornumpre = errornum;

        //if (!(k % 10))
        cout << "��" << k << "�ε������ع���" << errorNum(chi, I, J, K, R, U, V, W, a3) << endl;

    }


    cout << "�ֽ����" << endl;

}




//���ȱ�ʾ
Mat dizhi(vector<Mat> chi,int n,int R,Mat U,Mat V,Mat W) {
    Mat C;
    int I = chi[0].rows;
    int J = chi[0].cols;
    int K = n;

    C.create(I*J, R, CV_64F);

    //Khatri-Rao��  C
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


    cout << "���ȱ�ʾ���" << endl;
    return Q;


}


//����������
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
    int success = 0;//�ɹ�����

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

        cout << "chi_1���ڷֽ���:" << endl;
        CP_decomposition_own(im1, n, R, U1, V1, W1);
        cout << "chi_2���ڷֽ���:" << endl;
        CP_decomposition_own(im2, n, R, U2, V2, W2);

        Mat Q1 = dizhi(chi_1, n, R, U1, V1, W1);
        Mat Q2 = dizhi(chi_2, n, R, U2, V2, W2);

        float yuxian = cos_simple(Q1, Q2);

        cout << "��" + to_string(i) + "���ļ���������������Ϊ��" << yuxian;

        if (yuxian > 0.7)
            success++;
    }

    cout << "ʶ���ʣ�" << success / 40<<endl;
















    //Mat src;
    //    src=imread("./image_change/s26/1.jpg");//����ͼƬ
    //Mat src_gray, LBP_src, MB_LBPsrc;
    //if (src.empty())  //�ж�ͼƬ�Ƿ�Ϊ��
    //{
    //    cout << "could not load image1" << endl;
    //    return -1;
    //}

    //imshow("123", src);

    //cvtColor(src, src_gray, COLOR_BGR2GRAY);//ת��Ϊ�Ҷ�ͼ
    //imshow("output", src_gray);//����Ҷ�ͼ

    //getOriginLBPFeature<uchar>(src_gray, LBP_src);
    //imshow("LBP", LBP_src);

    //getMultiScaleBlockLBPFeature(src_gray, MB_LBPsrc, 9);
    //imshow("MB_LBP_9", MB_LBPsrc);

    //getMultiScaleBlockLBPFeature(src_gray, MB_LBPsrc, 15);
    //imshow("MB_LBP_15", MB_LBPsrc);

    //getMultiScaleBlockLBPFeature(src_gray, MB_LBPsrc, 21);
    //imshow("MB_LBP_21", MB_LBPsrc);


    //waitKey(0); //�������������һֱ��ʾ
    return 0;
}



