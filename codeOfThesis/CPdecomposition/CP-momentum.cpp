#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>
using namespace std;
using namespace cv;




vector<Mat> CP_decomposition() {
	//读入图片形成张量
	Mat src;
	vector<Mat> chi;
	for (int j = 0; j < 4; j++) {
		String inurl = "./orl_chlight/s1"+ '_' + to_string(j) + ".jpg";
		src = imread(inurl);//读入图片
		if (src.empty())  //判断图片是否为空
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








