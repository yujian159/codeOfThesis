#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>
using namespace std;
using namespace cv;




vector<Mat> CP_decomposition() {
	//����ͼƬ�γ�����
	Mat src;
	vector<Mat> chi;
	for (int j = 0; j < 4; j++) {
		String inurl = "./orl_chlight/s1"+ '_' + to_string(j) + ".jpg";
		src = imread(inurl);//����ͼƬ
		if (src.empty())  //�ж�ͼƬ�Ƿ�Ϊ��
		{
			cout << "could not load image1" << endl;
			return (vector<Mat>)-1;
		}

		chi.push_back(src);
	}


	for (int i = 0; i < 4; i++) {
		imshow("tu", chi[i]);
	}

}








int main() {
	CP_decomposition();






	return 0;
}








