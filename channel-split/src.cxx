#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
using namespace cv;
using namespace std;
 
int main() {
VideoCapture stream1(1);   //0 is the id of video device.0 if you have only one camera.
 
if (!stream1.isOpened()) { //check if video device has been initialised
cout << "cannot open camera";
}

int low_h = 80; int low_s = 50; int low_v = 50;
int high_h = 140; int high_s = 255; int high_v = 255;

SimpleBlobDetector::Params params;
params.minThreshold = 50;
params.maxThreshold = 150;

// Filter by Area.
params.filterByArea = true;
params.minArea = 1500;
params.maxArea = 150000;

// Filter by Circularity
params.filterByCircularity = false;
//params.minCircularity = 0.1;

// Filter by Convexity
params.filterByConvexity = false;
//params.minConvexity = 0.87;

// Filter by Inertia
params.filterByInertia = false;
//params.minInertiaRatio = 0.01;

params.filterByColor = false;



Ptr<SimpleBlobDetector> my_blob_detector = SimpleBlobDetector::create(params);
vector<KeyPoint> keypoints;

//unconditional loop
while (true) {

Mat cameraFrame;
Mat frame_threshold;
Mat frame_with_keypoints;
Mat hsv_frame;
stream1.read(cameraFrame);
GaussianBlur(cameraFrame, cameraFrame,Size(5,5), 0.23, 0,BORDER_DEFAULT );
cvtColor(cameraFrame,hsv_frame,CV_BGR2HSV);
inRange(hsv_frame,Scalar(low_h,low_s,low_v), Scalar(high_h,high_s,high_v),frame_threshold);
my_blob_detector->detect(frame_threshold, keypoints);

//detector.detect(cameraFrame, keypoints); //detect blobs
// Draw detected blobs as red circles.
// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures
// the size of the circle corresponds to the size of blob
drawKeypoints( cameraFrame, keypoints, frame_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

try{
Point2f point1=keypoints.at(0).pt;
float x1=point1.x;
float y1=point1.y;
cout<< "x1 = " << x1 << "\t" << "y1 = " << y1 << endl; 

}
catch(const std::out_of_range& oor)
{

cout<< "Object Missing\t" << endl;
goto LOOP;
}

LOOP:
imshow("cam", frame_threshold);
imshow("Blobs", frame_with_keypoints); 
if (waitKey(30) >= 0)
break;
}
return 0;
}
