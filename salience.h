#include <opencv2/opencv.hpp>
#include <keypoints/keypoints.h>
#include <opencv2/highgui/highgui.hpp>

std::vector<cv::Mat> createColourStack( cv::Mat& img, cv::Mat& centers );  //defining function color stack
// std::vector represents an array and can be accessed easily
std::vector<cv::Mat> createMotionStack(cv::Mat Get1, cv::VideoCapture stream1);
std::vector<cv::Mat> createTextureStack(cv::Mat& img, std::vector<vislab::keypoints::KPData>& datas );                            //defining create texture stack
std::vector<cv::Mat> createBlobKernels( std::vector<double> sigmas );  //defining createblobkernels
std::vector<cv::Mat> filterStack( const std::vector<cv::Mat>& inputstack, const std::vector<cv::Mat> filterStack );
//defining createblobkernels
cv::Mat sumstack(std::vector<cv::Mat> stack, std::vector<double> weights = std::vector<double>()); //defining sumstack
