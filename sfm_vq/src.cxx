#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

int main() {
    VideoCapture stream1(0);   //0 is the id of video device.0 if you have only one camera.
    
    if (!stream1.isOpened()) { //check if video device has been initialised
        cout << "cannot open camera device";
    }
    
    int minHessian = 400;
    Ptr<SURF> detector = SURF::create( minHessian );
    std::vector<KeyPoint> keypoints_1;
    
    
    //unconditional loop
    while (true) {
        Mat cameraFrame;
        Mat img_keypoints_1;
        stream1.read(cameraFrame);
        detector->detect( cameraFrame, keypoints_1 );
        drawKeypoints( cameraFrame, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
        imshow("cam", cameraFrame);
        imshow("features",img_keypoints_1);
        if (waitKey(30) >= 0)
            break;
    }
    return 0;
}

