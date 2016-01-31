
#include <keypoints/keypoints.h>
#include <salience/salience.h>
#include <iomanip>
#include <cstdlib>

using namespace std;
using namespace cv;
using namespace vislab::keypoints;

int main()
{
    Mat GetImg, image1, prvs,lab; //initialise the image parameters
    Size size(200,200);

    VideoCapture cap(0);   //0 is the id of video device.0 if you have only one camera

    vector<double> blobSigmas;

    blobSigmas.push_back(10);
    blobSigmas.push_back(20);
    blobSigmas.push_back(40);

    vector<Mat> blobKernels = createBlobKernels(blobSigmas);    //calling the create blob kernel function in main

    for(; ; )
    {
        if(!(cap.read(image1))) //get one frame form video
            break;

        GetImg = image1;

        //Resize
        resize(GetImg, prvs, size);
        resize(image1, image1, size);
        cvtColor(prvs, lab, CV_BGR2Lab);
        cvtColor(prvs, prvs, CV_BGR2GRAY);

        imshow("video",image1);

        // -------------------------Colours stack-----------------

        Mat centers;
        vector<Mat> colourStack = createColourStack( lab, centers);
        vector<Mat> colourBlobStack = filterStack( colourStack, blobKernels );
        Mat colourSaliency = sumstack(colourBlobStack);
        GaussianBlur(colourSaliency, colourSaliency, Size(21,21), 3.0);
        colourSaliency = colourSaliency/12;
        imshow("Color Saliency", colourSaliency);

        //-----------------------end color stack--------------------

        // -------------------------Motion stack-----------------

        vector<Mat> MotionStack = createMotionStack(prvs, cap);
        vector<Mat> MotionBlobStack = filterStack( MotionStack, blobKernels );
        Mat MotionSaliency = sumstack(MotionBlobStack);
        GaussianBlur(MotionSaliency, MotionSaliency, Size(21,21), 3.0);
        imshow("Motion Saliency", MotionSaliency);

        // -------------------------end Motion stack-----------------

        //----------------------------Texture--------------------------------------

//        vector<double> lambdas = makeLambdasLog(4,64,2);
//        vector<KPData> datas;
//        vector<KeyPoint> points = keypoints(gray,lambdas,datas,8,true);
//        vector<Mat> textureStack = createTextureStack( gray, datas );
//        vector<Mat> textureBlobStack = filterStack( textureStack, blobKernels );
//        Mat TextureSaliency = sumstack(textureBlobStack);
//        GaussianBlur(TextureSaliency, TextureSaliency, Size(21,21), 3.0);
//        //    TextureSaliency = TextureSaliency/7;
//        //    imshow("Texture Saliency", TextureSaliency);

        //----------------------------end Texture--------------------------------------

        if (waitKey(30) >= 0)
            break;
    }
    //    waitKey(0);
    return 0;
}

