#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/flann/dist.h>
#include <opencv2/highgui/highgui.hpp>
#include <keypoints/keypoints.h>
#include "opencv/cv.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv/highgui.h"
#include <salience/salience.h>

#define PI 3.1415

using namespace std;
using namespace cv;
using namespace vislab::keypoints;

//function sumstack
Mat sumstack( vector<Mat> stack, std::vector<double> weights)
{
    Mat result = Mat::zeros(stack[0].rows, stack[0].cols, stack[0].type()); //initialising the result to zeros of rows and cols

    if(weights.empty())
        for(unsigned i=0;i<stack.size();i++)
            weights.push_back(1./stack.size());

    if(weights.size() != stack.size())
    {
        std::cerr <<"Number of weights must match the number of images" << std::endl;
        return result;
    }

    for(unsigned i=0;i<stack.size(); i++)  //i is declared in order to access the stacks
    {
        Mat temp;
        threshold(stack[i], temp, 0, 0, THRESH_TOZERO);  //the value of stack[i] is stored in temp (thresholded)
        result = result + temp*weights[i]; //stack[i];  // the temp value is added to the result
//        result = result + temp; //stack[i];  // the temp value is added to the result
    }

    return result;
}

//function blobkernel
vector<Mat> createBlobKernels(vector<double> sigmas ) // Modified one
{
    vector<Mat> result; //calling the result of sumstack

    for(unsigned i=0; i<sigmas.size(); i++)     //initialising to access the sigmas
    {
        double sigma = sigmas[i];   //initialisig the value of sigma to sigma[i]
        int filtersize = sigma*4;
        if( filtersize % 2 == 0) filtersize++;        // Ensure that the kernel size is odd
        int offsetf = floor(filtersize/2);    //floor is used to round off to integer value

        Mat kernel( filtersize, filtersize, CV_32F );   //initialising the kernel and putting the filtersize into kernel
        Mat kernel_ex( filtersize, filtersize, CV_32F );   //initialising the kernel and putting the filtersize into kernel
        Mat kernel_inh( filtersize, filtersize, CV_32F );   //initialising the kernel and putting the filtersize into kernel
        float kernelsum = 0;

        // create the kernel we will be using (difference of gaussians
        for(int row=0; row<kernel.rows; row++)
        {
            for(int col=0; col<kernel.cols; col++)
            {   // return 0;
                double x = col - offsetf;
                double y = row - offsetf;

                double gauss1 = exp( -(x*x+y*y) / (sigma*sigma)   );
                double gauss2 = exp( -(x*x+y*y) / (sigma*sigma*4) );

//                kernel.at<float>(row,col) = gauss1 - gauss2;
//                kernelsum += gauss1 - gauss2;
                kernel_ex.at<float>(row,col) = gauss1;
                kernel_inh.at<float>(row,col) = gauss2;

            }
        }
        // Remove the DC component of the even kernel
//        kernel -= kernelsum/(filtersize*filtersize);
        kernel_ex *= 1/cv::sum(kernel_ex).val[0];
        kernel_inh *= 1/cv::sum(kernel_inh).val[0];
        kernel = kernel_ex-kernel_inh;

        result.push_back(kernel);
    }
    return result;
}


//---------------for color stack------------------------

vector<Mat> createColourStack( Mat& img, Mat& centers)
{
    vector<Mat> result;

    Mat palette;
    centers.convertTo(palette,CV_8UC3,1.0);
    int rows = img.rows; // /scalefactor;
    int cols = img.cols; // /scalefactor;

    for(unsigned i=0; i<6; i++)
        result.push_back(Mat::zeros(rows, cols, CV_32F));

    for(int colr=0; colr<result.size(); colr++)
    {
        for(int y=0; y<rows; y++)
            for(int x=0; x<cols; x++)
                switch(colr) {
                    case 0: result[colr].at<float>(y,x) =     img.at<Vec3b>(y,x)[0]/6; break;
                    //vec3b is called for 3 channels BGR [0] for blue
                    case 1: result[colr].at<float>(y,x) = 255-img.at<Vec3b>(y,x)[0]/6; break; //white
                    case 2: result[colr].at<float>(y,x) =     img.at<Vec3b>(y,x)[1]; break;
                    //[1] is for green
                    case 3: result[colr].at<float>(y,x) = 255-img.at<Vec3b>(y,x)[1]; break; //yellow
                    case 4: result[colr].at<float>(y,x) =     img.at<Vec3b>(y,x)[2]; break;
                    //[2] is for red
                    case 5: result[colr].at<float>(y,x) = 255-img.at<Vec3b>(y,x)[2]; break; //black
                     }
    }
    return result;
  }


//vector<Mat> createColourStack( Mat& img, Mat& centers) // Modified
//{
//    vector<Mat> result;

//    Mat palette;
//    centers.convertTo(palette,CV_8UC3,1.0);
//    int rows = img.rows; // /scalefactor;
//    int cols = img.cols; // /scalefactor;

//    for(unsigned i=0; i<3; i++)
//        result.push_back(Mat::zeros(rows, cols, CV_32F));

////    for(int colr=0; colr<result.size(); colr++)
////    {
////        for(int y=0; y<rows; y++)
////            for(int x=0; x<cols; x++)
////                switch(colr) {
////                    case 0: result[colr].at<float>(y,x) = img.at<Vec3b>(y,x)[0] + img.at<Vec3b>(y,x)[1] + img.at<Vec3b>(y,x)[2]; break;
////                    //vec3b is called for 3 channels BGR [0] for blue
////                    case 1: result[colr].at<float>(y,x) = img.at<Vec3b>(y,x)[2] - img.at<Vec3b>(y,x)[1]; break;
////                    //[1] is for green
////                    case 2: result[colr].at<float>(y,x) = img.at<Vec3b>(y,x)[0] - (img.at<Vec3b>(y,x)[1] + img.at<Vec3b>(y,x)[2])/2.f; break;
////                    //[2] is for red
////                     }
////    }

//    Mat converted;
//    img.convertTo(converted, CV_32FC3);

//    vector<Mat> planes_bgr;
//    split(converted, planes_bgr);

//    result[0] = planes_bgr[0] + planes_bgr[1] + planes_bgr[2];
//    result[0] /= 3*255.f;
////    result[0] /= 3.f;

//    result[1] = planes_bgr[2] - planes_bgr[1];
//    result[1] /= 255.f;

//    result[2] = planes_bgr[0] - (planes_bgr[1] + planes_bgr[2])/2.f;
//    result[2] /= 255.f;

////    imshow("plane0",result[0]);
////    imshow("plane1",result[1]);
////    imshow("plane2",result[2]);
////    result[0] /= 3*255.f;
////    result[1] /= 255.f;
////    result[2] /= 255.f;

//    return result;
//  }

//--------------------------end color stack-----------------------------

vector<Mat> filterStack( const vector<Mat>& inputstack, const vector<Mat> filterstack )
{
    vector<Mat> result;

    for(unsigned i=0; i<inputstack.size(); i++)
    {
        Mat response, tempresponse;
        response = Mat::zeros(inputstack[i].rows, inputstack[i].cols, CV_32F);

        for(unsigned j=0; j<filterstack.size(); j++)
        {
            Mat temp = inputstack[i];
             GaussianBlur(inputstack[i], temp, Size(21,21), 4.0);
//             cout<<"loop.....\n";
            filter2D( temp, tempresponse, inputstack[i].depth(), filterstack[j]);
//            filter2D( temp, tempresponse, response.type(), filterstack[j]);
//            cout<<"loopend....\n";
            response += tempresponse;
        }
        result.push_back(response);
    }
    return result;
}

 //-----------------------------for Motion stack-----------------------------------------------

 vector<Mat> createMotionStack(Mat Get1, VideoCapture stream1)
 {

  vector<Mat> result;
  Size size(200,200);
  Mat prvs, next, cflow, GetImg; //current frame
  prvs = Get1;
  Mat flow(prvs.rows, prvs.cols, CV_32FC2);
  int r = flow.rows;
  int c = flow.cols;
  Mat Mag(Size(r, c), CV_32FC1);
  Mat angle(Size(r, c), CV_32F);

  for(;;)
  {

      if(!(stream1.read(GetImg))) //get one frame form video
          break;
      //Resize
//      next = GetImg;
      resize(GetImg, next,size);
//       resize(GetImg, next, Size(), 0.25, 0.25, INTER_LINEAR);
      cvtColor(next, next, CV_BGR2GRAY);

   calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0); // Optical flow for every pixel

   cvtColor(prvs, cflow, CV_GRAY2BGR);

   for (int x=0; x<flow.rows; x++)
   {
       for (int y=0; y<flow.cols; y++)
       {
           cv::Vec2f flowXY = flow.at<cv::Vec2f>(x,y);
           float K1 = flowXY[0];
           float K2 = flowXY[1];

           Mag.at<float>(y,x) = sqrt(pow(K2, 2) + pow(K1, 2));
           angle.at<float>(y,x) = atan2 (K2,K1) * 180.0 / PI;

       }
   }
   break;
  }

  //------------------ the changes---------------------

//  for(unsigned i=0; i<11; i++)
//      result.push_back(Mat::zeros(prvs.rows, prvs.cols, CV_32F));

//  // temp matrices
//   cv::Mat temp, temp2, high_mag;

//   // fast movements
//   cv::threshold(Mag, high_mag, 5, 1, THRESH_BINARY);

//   // angles
//   // double threshold for each interval, point-wise multiply with fast magnitude matrix
//   cv::threshold(angle, temp, 0, 1, THRESH_BINARY);
//   cv::threshold(angle, temp2, 45, 1, THRESH_BINARY_INV);
//   result[0] = temp.mul(temp2).mul(high_mag);
//   cv::threshold(angle, temp, 45, 1, THRESH_BINARY);
//   cv::threshold(angle, temp2, 90, 1, THRESH_BINARY_INV);
//   result[1] = temp.mul(temp2).mul(high_mag);
//   cv::threshold(angle, temp, 90, 1, THRESH_BINARY);
//   cv::threshold(angle, temp2, 135, 1, THRESH_BINARY_INV);
//   result[2] = temp.mul(temp2).mul(high_mag);
//   cv::threshold(angle, temp, 135, 1, THRESH_BINARY);
//   cv::threshold(angle, temp2, 180, 1, THRESH_BINARY_INV);
//   result[3] = temp.mul(temp2).mul(high_mag);
//   cv::threshold(angle, temp, -180, 1, THRESH_BINARY);
//   cv::threshold(angle, temp2, -135, 1, THRESH_BINARY_INV);
//   result[4] = temp.mul(temp2).mul(high_mag);
//   cv::threshold(angle, temp, -135, 1, THRESH_BINARY);
//   cv::threshold(angle, temp2, -90, 1, THRESH_BINARY_INV);
//   result[5] = temp.mul(temp2).mul(high_mag);
//   cv::threshold(angle, temp, -90, 1, THRESH_BINARY);
//   cv::threshold(angle, temp2, -45, 1, THRESH_BINARY_INV);
//   result[6] = temp.mul(temp2).mul(high_mag);
//   cv::threshold(angle, temp, -45, 1, THRESH_BINARY);
//   cv::threshold(angle, temp2, 0, 1, THRESH_BINARY_INV);
//   result[7] = temp.mul(temp2).mul(high_mag);

//   // magnitudes
//   cv::threshold(Mag, temp, 0, 1, THRESH_BINARY);
//   cv::threshold(Mag, temp2, 5, 1, THRESH_BINARY_INV);
//   result[8] = temp.mul(temp2);
//   cv::threshold(Mag, temp, 5, 1, THRESH_BINARY);
//   cv::threshold(Mag, temp2, 10, 1, THRESH_BINARY_INV);
//   result[9] = temp.mul(temp2);
//   cv::threshold(Mag, temp, 10, 1, THRESH_BINARY);
//   cv::threshold(Mag, temp2, 150, 1, THRESH_BINARY_INV);
//   result[10] = temp.mul(temp2);

  //--------------------end changes---------------------

//  for(unsigned i=0; i<11; i++)
//      result.push_back(Mat::zeros(prvs.rows, prvs.cols, CV_32F));

//  for(int ch=0; ch<result.size(); ch++)
//  {
//      for(int y=0; y<prvs.rows; y++)
//          for(int x=0; x<prvs.cols; x++)
//          {
//              bool fast = true;
////              if( Mag.at<float>(y,x) < 1) fast = false;
//              switch(ch)
//              {
//              case 0: result[ch].at<float>(y,x) = Mag.at<float>(y,x) > 0 && Mag.at<float>(y,x) < 5 ? 1 : 0 ; break;
//              case 1: result[ch].at<float>(y,x) = Mag.at<float>(y,x) > 5 && Mag.at<float>(y,x) < 10 ? 1 : 0 ; break;
//              case 2: result[ch].at<float>(y,x) = Mag.at<float>(y,x) > 10 && Mag.at<float>(y,x) < 15 ? 1 : 0 ; break;
//              case 3: result[ch].at<float>(y,x) = angle.at<float>(y,x) > 0 && angle.at<float>(y,x) < 45 && fast ? 1 : 0 ; break;
//              case 4: result[ch].at<float>(y,x) = angle.at<float>(y,x) > 45 && angle.at<float>(y,x) < 90 && fast ? 1 : 0 ; break;
//              case 5: result[ch].at<float>(y,x) = angle.at<float>(y,x) > 90 && angle.at<float>(y,x) < 135 && fast ? 1 : 0 ; break;
//              case 6: result[ch].at<float>(y,x) = angle.at<float>(y,x) > 135 && angle.at<float>(y,x) < 180 && fast ? 1 : 0 ; break;
//              case 7: result[ch].at<float>(y,x) = angle.at<float>(y,x) > 180 && angle.at<float>(y,x) < 225 && fast ? 1 : 0 ; break;
//              case 8: result[ch].at<float>(y,x) = angle.at<float>(y,x) > 225 && angle.at<float>(y,x) < 270 && fast ? 1 : 0 ; break;
//              case 9: result[ch].at<float>(y,x) = angle.at<float>(y,x) > 270 && angle.at<float>(y,x) < 315 && fast ? 1 : 0 ; break;
//              case 10: result[ch].at<float>(y,x) = angle.at<float>(y,x) > 315 && angle.at<float>(y,x) < 360 && fast ? 1 : 0 ; break;
//              }
//          }
//  }

  for(unsigned i=0; i<11; i++)
      result.push_back(Mat::zeros(prvs.rows, prvs.cols, CV_32F));

  for(int ch=0; ch<result.size(); ch++)
  {
      for(int y=0; y<prvs.rows; y++)
          for(int x=0; x<prvs.cols; x++)
          {
              bool fast = true;
              if( Mag.at<float>(y,x) < 3) fast = false;
              switch(ch)
              {
//              case 0: result[ch].at<float>(y,x) = Mag.at<float>(y,x) > 0 && Mag.at<float>(y,x) < 5  && fast ? Mag.at<float>(y,x) : 0 ; break;
//              case 1: result[ch].at<float>(y,x) = Mag.at<float>(y,x) > 5 && Mag.at<float>(y,x) < 10  && fast ? Mag.at<float>(y,x) : 0 ; break;
//              case 2: result[ch].at<float>(y,x) = Mag.at<float>(y,x) > 10 && Mag.at<float>(y,x) < 15  && fast ? Mag.at<float>(y,x) : 0 ; break;
              case 3: result[ch].at<float>(y,x) = angle.at<float>(y,x) > 0 && angle.at<float>(y,x) < 45 && fast ? angle.at<float>(y,x) : 0 ; break;
              case 4: result[ch].at<float>(y,x) = angle.at<float>(y,x) > 45 && angle.at<float>(y,x) < 90 && fast ? angle.at<float>(y,x) : 0 ; break;
              case 5: result[ch].at<float>(y,x) = angle.at<float>(y,x) > 90 && angle.at<float>(y,x) < 135 && fast ? angle.at<float>(y,x) : 0 ; break;
              case 6: result[ch].at<float>(y,x) = angle.at<float>(y,x) > 135 && angle.at<float>(y,x) < 180 && fast ? angle.at<float>(y,x) : 0 ; break;
              case 7: result[ch].at<float>(y,x) = angle.at<float>(y,x) > 180 && angle.at<float>(y,x) < 225 && fast ? angle.at<float>(y,x) : 0 ; break;
              case 8: result[ch].at<float>(y,x) = angle.at<float>(y,x) > 225 && angle.at<float>(y,x) < 270 && fast ? angle.at<float>(y,x) : 0 ; break;
              case 9: result[ch].at<float>(y,x) = angle.at<float>(y,x) > 270 && angle.at<float>(y,x) < 315 && fast ? angle.at<float>(y,x) : 0 ; break;
              case 10: result[ch].at<float>(y,x) = angle.at<float>(y,x) > 315 && angle.at<float>(y,x) < 360 && fast ? angle.at<float>(y,x) : 0 ; break;
              }
          }
  }
//  cout<<" Mag = "<<Mag.at<float>(5,5)<<"\n";
//  cout<<"Angle = "<<angle.at<float>(5,5)<<"\n";
  return result;
 }

 //-----------------------------end motion stack-----------------------------------------------

 //-----------------------------Texture stack-----------------------------------------------
 vector<Mat> createTextureStack(Mat& img, vector<KPData>& datas)
 {
     vector<Mat> result, result1;
     Mat spec,spec1;
     Mat sumrows, sumcols;
     Point minLoc_r, maxLoc_r, minLoc_c, maxLoc_c;
     double max_val_o, max_val_f;

     if(datas.empty())
         return result;
     int freqs = datas.size();
     int oris  = datas[0].C_array.size();
     int rows = img.rows;
     int cols = img.cols;

      Mat Std_o(rows, cols, CV_32F);
      Mat Std_f(rows, cols, CV_32F);
      Mat Mu_o(rows, cols, CV_32F);
      Mat Mu_f(rows, cols, CV_32F);
      Mat eps(rows, cols, CV_32F);

     vector<int> pyrscales;
     for(unsigned i=0; i<datas.size(); i++)
         pyrscales.push_back(round(float(img.rows)/float(datas[i].C_array[0].rows)));

     for(unsigned i=0; i<4; i++)
         result.push_back(Mat::zeros(rows, cols, CV_32F));

     for(unsigned i=0; i<2; i++)
         result1.push_back(Mat::zeros(rows, cols, CV_32F));

     for(int y=0; y<rows; y++)
         for(int x=0; x<cols; x++)
             {
             Mat spectrum(freqs, oris, CV_32F);
             for(int sy=0; sy<spectrum.rows; sy++)
                 for(int sx=0; sx<spectrum.cols; sx++)
                     spectrum.at<float>(sy,sx) = datas[sy].C_array[sx](y/pyrscales[sy],x/pyrscales[sy]);

//             cout<<"ty = "<<img.type()<<"\n";
//             if(x==211)
//                 if(y==210)
//                 {
//                     imwrite("/home/saikrishna/Desktop/GCPR/insidespectrumwhite.png",spectrum);
//                     Scalar intensity = img.at<uchar>(y, x);
////                     Vec3b intensity1 = img.at<Vec3b>(y, x);
////                     uchar blue = intensity1.val[0];
////                     uchar green = intensity1.val[1];
////                     uchar red = intensity1.val[2];
//                     cout<<"blue = "<<intensity.val[0]<<"\n";
//                     cout<<"green = "<<intensity.val[1]<<"\n";
////                     cout<<"red = "<<intensity.val[2]<<"\n";
////                     cout<<int(blue)<<"\n";
////                     cout<<int(green)<<"\n";
////                     cout<<int(red)<<"\n";
//                     cout<<"=====================\n";
//                 }

             blur(spectrum, spec, Size(3, 3));
             spec1=spec/3;
             reduce( spec1, sumrows, 0, CV_REDUCE_AVG);
             reduce( spec1, sumcols, 1, CV_REDUCE_AVG );

             minMaxLoc(sumrows, NULL, &max_val_f, &minLoc_r, &maxLoc_r); // rows
             minMaxLoc(sumcols, NULL, &max_val_o, &minLoc_c, &maxLoc_c); // cols

             int r=sumrows.rows;
             int c=sumrows.cols;

             float v1, v2, Mu=0, Mu1, val=0, var, stddev;
             float v12, v22, Mu2=0, Mu12, val2=0, var2, stddev2;
             float f1, out, stddevr, total = 0;
             float fc, outc,stddevc, totalc=0;

             for (int i1=0; i1<r; i1++)
                 for (int j1=0; j1<c; j1++)
                 {
                     v1 = sumrows.at<float>(j1);
                     Mu = Mu + v1;
                 }
             Mu1 = Mu/c;

             for (int i2=0; i2<r; i2++)
                 for (int j2=0; j2<c; j2++)
                 {
                     v2 = pow((sumrows.at<float>(j2)-Mu1),2);
                     val = val + v2;
                 }
             var = val/c;
             stddev = sqrt(var);

//             float kk = 0, kk1 = 0, k1;

//             for(int a=0; a<r; a++)
//                 {
//                 for(int b=0;b<c;b++)
//                     {
//                     f1 = sumrows.at<float>(b);
//                     kk = kk + f1;
//                 }
//             }

//             for(int a=0; a<r; a++)
//                 {
//                 for(int b=0;b<c;b++)
//                     {
//                     k1 = sumrows.at<float>(b)/(kk);
//                     kk1 = kk1 + k1;
//                     out = (k1* (pow((b-maxLoc_r.x),2)));
//                     total = (total + out);
//                 }
//             }
//             total = fabs(total);
//             stddevr = sqrt(total);

             Std_o.at<float>(y,x) = stddev;  // Method 1
             Mu_o.at<float>(y,x) = Mu1;

//             ort.at<float>(y,x) = stddevr;  // Method 2
//             Mu_o.at<float>(y,x) = max_val_o;

             int ro =sumcols.rows;
             int co=sumcols.cols;

             for (int i12=0; i12<ro; i12++)
                 for (int j12=0; j12<co; j12++)
                 {
                     v12 = sumrows.at<float>(i12);
                     Mu2 = Mu2 + v12;
                 }
             Mu12 = Mu2/ro;

             for (int i22=0; i22<ro; i22++)
                 for (int j22=0; j22<co; j22++)
                 {
                     v22 = pow((sumrows.at<float>(i22)-Mu12),2);
                     val2 = val2 + v22;
                 }
             var2 = val2/ro;
             stddev2 = sqrt(var2);

//             float sk = 0, kkk1 = 0, sk1;

//             for(int c=0; c<ro; c++)
//                 {
//                 for(int d=0;d<co;d++)
//                     {
//                     fc = sumrows.at<float>(c);
//                     sk = sk + fc;
//                 }
//             }

//             for(int m=0; m<ro; m++)
//                 {
//                 for(int n=0;n<co;n++)
//                     {
//                     sk1 = sumrows.at<float>(m)/(sk);
//                     kkk1 = kkk1 + sk1;
//                     outc = (sk1* (pow((m-maxLoc_c.y),2)));
//                     totalc = totalc + outc;
//                 }
//             }

//             totalc = fabs(totalc);
//             stddevc = sqrt(totalc);

             Std_f.at<float>(y,x) = stddev2; // Method 1
             Mu_f.at<float>(y,x) = Mu12;

//             fre.at<float>(y,x) = stddevc; // Method 2
//             Mu_f.at<float>(y,x) = max_val_f;
//             eps.at<float>(y,x) = (stddev2-stddev)/(stddev2+stddev);
//             if(eps.at<float>(y,x)>0.9)
//                 cout<<"hi = "<<eps.at<float>(y,x)<<"\n";

        }

     for(int ch=0; ch<result.size(); ch++)
         {
              for(int y=0; y<rows; y++)
                  for(int x=0; x<cols; x++)
                  {
                      switch(ch)
                      {
                      case 0: result[ch].at<float>(y,x) = Std_o.at<float>(y,x)>0 && Std_o.at<float>(y,x)<100 ? std::abs(Std_o.at<float>(y,x)) : 0 ; break;
                      case 1: result[ch].at<float>(y,x) = Std_f.at<float>(y,x)>0 && Std_f.at<float>(y,x)<100 ? std::abs(Std_f.at<float>(y,x)) : 0 ;break;
                      case 2: result[ch].at<float>(y,x) = Mu_o.at<float>(y,x)>0 && Mu_o.at<float>(y,x)<100 ? std::abs(Mu_o.at<float>(y,x)) : 0 ;break;
                      case 3: result[ch].at<float>(y,x) = Mu_f.at<float>(y,x)>0 && Mu_f.at<float>(y,x)<100 ? std::abs(Mu_f.at<float>(y,x)) : 0 ;break;
                      }
                  }
              //              int top, bottom, left, right;
              //              Mat src, dst;
              //              src = result[ch];
              //              dst = result[ch];
              //              top = (int) (0.10*src.rows); bottom = (int) (0.10*src.rows);
              //              left = (int) (0.10*src.cols); right = (int) (0.10*src.cols);

              //              copyMakeBorder( src, result[ch], top, bottom, left, right, BORDER_REPLICATE );
         }

//     result[0] = Std_o;
//     result[1] = Std_f;
//     result[2] = Mu_o;
//     result[3] = Mu_f;

//     Mat leo,leo1,leo2,leo3;
//     result[0].convertTo(leo, CV_8UC1);
//     result[1].convertTo(leo1, CV_8UC1);
//     result[2].convertTo(leo2, CV_8UC1);
//     result[3].convertTo(leo3, CV_8UC1);

//     imshow("result0", leo);
//     imshow("result1", leo1);
//     imshow("result2", leo2);
//     imshow("result3", leo3);

//     imshow("Orientation STD", result[0]/255.f);
//     imshow("Frequency STD", result[1]/255.f);
//     imshow("Orientation Mean", result[2]/255.f);
//     imshow("Frequency Mean", result[3]/255.f);


//     Mat leo,leo1,leo2,leo3;
//     result[0].convertTo(leo, CV_8UC1);
//     result[1].convertTo(leo1, CV_8UC1);
//     result[2].convertTo(leo2, CV_8UC1);
//     result[3].convertTo(leo3, CV_8UC1);

//     Mat src = result[0];
//     double minVal, maxVal;
//     Point min_loc, max_loc;
//     minMaxLoc(src, &minVal, &maxVal, &min_loc, &max_loc);
//     Mat res = src;
//     res = res/maxVal;
//     res = res*255;
//     Mat src1 = result[1];
//     double minVal1, maxVal1;
//     Point min_loc1, max_loc1;
//     minMaxLoc(src1, &minVal1, &maxVal1, &min_loc1, &max_loc1);
//     Mat res1 = src1;
//     res1 = res1/maxVal1;
//     res1 = res1*255;
//     Mat src2 = result[2];
//     double minVal2, maxVal2;
//     Point min_loc2, max_loc2;
//     minMaxLoc(src2, &minVal2, &maxVal2, &min_loc2, &max_loc2);
//     Mat res2 = src2;
//     res2 = res2/maxVal2;
//     res2 = res2*255;
//     Mat src3 = result[3];
//     double minVal3, maxVal3;
//     Point min_loc3, max_loc3;
//     minMaxLoc(src3, &minVal3, &maxVal3, &min_loc3, &max_loc3);
//     Mat res3 = src3;
//     res3 = res3/maxVal3;
//     res3 = res3*255;
//     double minVal4, maxVal4;
//     Point min_loc4, max_loc4;
//     minMaxLoc(res3, &minVal4, &maxVal4, &min_loc4, &max_loc4);
//     cout<<"low = "<<minVal4<<" high = "<<maxVal4<<"\n";

//     imwrite("/home/saikrishna/Desktop/Feature Maps2/0_1_1650_Orientation STD.png",res);
//     imwrite("/home/saikrishna/Desktop/Feature Maps2/0_1_1650_Frequency STD.png",res1);
//     imwrite("/home/saikrishna/Desktop/Feature Maps2/0_1_1650_Orientation Mean.png",res2);
//     imwrite("/home/saikrishna/Desktop/Feature Maps2/0_1_1650_Frequency Mean.png",res3);
//     imshow("result0", leo);
//     imshow("result1", leo1);
//     imshow("result2", leo2);
//     imshow("result3", leo3);

    return result;
 }
 //-----------------------------Enf of Texture stack-----------------------------------------------
