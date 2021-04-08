#include "opencv4/opencv2/opencv.hpp"
#include <cstdio>
#include <opencv2/imgproc.hpp>
#include <opencv4/opencv2/core/hal/interface.h>
#include <opencv4/opencv2/xfeatures2d.hpp>
#include <fstream>
#include <bitset>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

// Perform a single thinning iteration, which is repeated until the skeletization is finalized
void thinningIteration(Mat& im, int iter)
{
    Mat marker = Mat::zeros(im.size(), CV_8UC1);
    for (int i = 1; i < im.rows-1; i++)
    {
        for (int j = 1; j < im.cols-1; j++)
        {
            uchar p2 = im.at<uchar>(i-1, j);
            uchar p3 = im.at<uchar>(i-1, j+1);
            uchar p4 = im.at<uchar>(i, j+1);
            uchar p5 = im.at<uchar>(i+1, j+1);
            uchar p6 = im.at<uchar>(i+1, j);
            uchar p7 = im.at<uchar>(i+1, j-1);
            uchar p8 = im.at<uchar>(i, j-1);
            uchar p9 = im.at<uchar>(i-1, j-1);

            int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                     (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                     (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                     (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
            int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
            int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
            int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

            if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                marker.at<uchar>(i,j) = 1;
        }
    }

    im &= ~marker;
}

void createCrossingNumber(Mat& im, Mat& out){
    Mat marker = Mat::zeros(im.size(), CV_8UC1 );
    for (int i=1;i<im.rows-1;i++){
        for (int j=1; j<im.cols-1 ; j++){
            uchar p2 = im.at<uchar>(i-1, j);
            uchar p3 = im.at<uchar>(i-1, j+1);
            uchar p4 = im.at<uchar>(i, j+1);
            uchar p5 = im.at<uchar>(i+1, j+1);
            uchar p6 = im.at<uchar>(i+1, j);
            uchar p7 = im.at<uchar>(i+1, j-1);
            uchar p8 = im.at<uchar>(i, j-1);
            uchar p9 = im.at<uchar>(i-1, j-1);

            if (im.at<uchar>(i,j)==1){
                marker.at<uchar>(i,j)=(abs(p2-p3)+abs(p3-p4)+abs(p4-p5)+abs(p5-p6)+abs(p6-p7)+abs(p7-p8)+abs(p8-p9)+abs(p9-p2))/2;
                //printf("value of marker i j : %d %d %d\n",marker.at<uchar>(i,j), i , j);
            }
            else{
                marker.at<uchar>(i,j)=0;
            }
        }
    }
    out=marker;

}

// Function for thinning any given binary image within the range of 0-255. If not you should first make sure that your image has this range preset and configured!
void thinning(Mat& im)
{
	 // Enforce the range tob e in between 0 - 255
    im /= 255;

    Mat prev = Mat::zeros(im.size(), CV_8UC1);
    Mat diff;

    do {
        thinningIteration(im, 0);
        thinningIteration(im, 1);
        absdiff(im, prev, diff);
        im.copyTo(prev);
    }
    while (countNonZero(diff) > 0);

    im *= 255;
}



double compareWithHaris(char* nom1, char* nom2, int harrisThreshold, double radiusSize){

    Mat input = imread(nom1, IMREAD_GRAYSCALE);
    if(input.empty()){cerr << "Image not read correctly. Check if path is correct!" << endl;}

    Mat input_binary;
    threshold(input, input_binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
    Mat input_thinned = input_binary.clone();
    thinning(input_thinned);
    Mat harris_corners, harris_normalised;
    harris_corners = Mat::zeros(input_thinned.size(), CV_8UC1);
    cornerHarris(input_thinned, harris_corners, 2, 3, 0.04, BORDER_DEFAULT);
    normalize(harris_corners, harris_normalised, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

    vector<KeyPoint> keypoints;
    Mat rescaled;
    convertScaleAbs(harris_normalised, rescaled);
    Mat harris_c(rescaled.rows, rescaled.cols, CV_8UC3);
    Mat in[] = { rescaled, rescaled, rescaled };
    int from_to[] = { 0,0, 1,1, 2,2 };
    mixChannels( in, 3, &harris_c, 1, from_to, 3 );
    for(int x=0; x<harris_normalised.cols; x++){
        for(int y=0; y<harris_normalised.rows; y++){
            if ( (int)harris_normalised.at<float>(y, x) > harrisThreshold ){
                circle(harris_c, Point(x, y), 5, Scalar(0,255,0), 1);
                circle(harris_c, Point(x, y), 1, Scalar(0,0,255), 1);
                keypoints.push_back( KeyPoint (x, y, 1) );
            }
        }
    }
    Ptr<Feature2D> orb_descriptor = ORB::create();
    Mat descriptors;
    orb_descriptor->compute(input_thinned, keypoints, descriptors);


    Mat input2 = imread(nom2, IMREAD_GRAYSCALE);
    Mat input_binary2;
    threshold(input2, input_binary2, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
    Mat input_thinned2 = input_binary2.clone();
    thinning(input_thinned2);
    Mat harris_corners2, harris_normalised2;
    harris_corners2 = Mat::zeros(input_thinned2.size(), CV_32FC1);
    cornerHarris(input_thinned2, harris_corners2, 2, 3, 0.04, BORDER_DEFAULT);
    normalize(harris_corners2, harris_normalised2, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    vector<KeyPoint> keypoints2;
    Mat rescaled2;
    convertScaleAbs(harris_normalised2, rescaled2);
    Mat harris_c2(rescaled2.rows, rescaled2.cols, CV_8UC3);
    Mat in2[] = { rescaled2, rescaled2, rescaled2 };
    int from_to2[] = { 0,0, 1,1, 2,2 };
    mixChannels( in2, 3, &harris_c2, 1, from_to2, 3 );
    for(int x=0; x<harris_normalised2.cols; x++){
        for(int y=0; y<harris_normalised2.rows; y++){
            if ( (int)harris_normalised2.at<float>(y, x) > harrisThreshold ){
                circle(harris_c2, Point(x, y), 5, Scalar(0,255,0), 1);
                circle(harris_c2, Point(x, y), 1, Scalar(0,0,255), 1);
                keypoints2.push_back( KeyPoint (x, y, 1) );
          }
       }
    }
    Mat descriptors2;
    orb_descriptor->compute(input_thinned2, keypoints2, descriptors2);


    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    vector<vector< DMatch >> matches;
    matcher->radiusMatch(descriptors, descriptors2, matches,radiusSize);

    int score = 0;
    for(int i=0; i < matches.size(); i++){
        for (int j=0;j<matches[0].size();j++){
            DMatch current_match = matches[i][j];
            //printf("idTRAIN %d, idQUERRY %d, score %f\n",current_match.trainIdx,current_match.queryIdx,current_match.distance);
            if (current_match.distance>1e-3){score++;}
            //score++;
        }
    }
    //cerr << endl << "Current matching score = " << score << endl;
    printf("Harris\t%d\t%s\t%s\t%d\t%d\t%d\t%f\n",harrisThreshold,nom1,nom2,(int)keypoints.size(),(int)keypoints2.size(),score,radiusSize);
    return (score);

}

double compareWithCN(char* nom1, char* nom2, double radiusSize, bool samePerson){

    Mat input = imread(nom1, IMREAD_GRAYSCALE);
    if(input.empty()){
	    cerr << "Image not read correctly. Check if path is correct!" << endl;
    }
    Mat input_binary;
    threshold(input, input_binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
    Mat input_thinned = input_binary.clone();
    thinning(input_thinned);
    Mat harris_corners, harris_normalised;
    harris_corners = Mat::zeros(input_thinned.size(), CV_8UC1);
    Mat crossingNumber;
    threshold(input_thinned, crossingNumber, 0, 1, THRESH_BINARY | THRESH_OTSU);
    createCrossingNumber(crossingNumber, harris_corners);
    normalize(harris_corners, harris_normalised, 0, 255, NORM_MINMAX, CV_8UC1, Mat());
    vector<KeyPoint> keypoints;
    Mat rescaled;
    convertScaleAbs(harris_normalised, rescaled);
    Mat harris_c(rescaled.rows, rescaled.cols, CV_8UC3);
    Mat in[] = { rescaled, rescaled, rescaled };
    int from_to[] = { 0,0, 1,1, 2,2 };
    mixChannels( in, 3, &harris_c, 1, from_to, 3 );
    for(int x=0; x<harris_corners.cols; x++){
        for(int y=0; y<harris_corners.rows; y++){
            if ((int)harris_corners.at<uchar>(y,x)==3 || (int)harris_corners.at<uchar>(y,x)==1){
                circle(harris_c, Point(x, y), 5, Scalar(0,255,0), 1);
                circle(harris_c, Point(x, y), 1, Scalar(0,0,255), 1);
                keypoints.push_back( KeyPoint (x, y, 1) );
            }
        }
    }
    Ptr<Feature2D> orb_descriptor = ORB::create();
    Mat descriptors;
    orb_descriptor->compute(input_thinned, keypoints, descriptors);




    Mat input2 = imread(nom2, IMREAD_GRAYSCALE);
    Mat input_binary2;
    threshold(input2, input_binary2, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
    Mat input_thinned2 = input_binary2.clone();
    thinning(input_thinned2);
    Mat harris_corners2, harris_normalised2;
    harris_corners2 = Mat::zeros(input_thinned2.size(), CV_32FC1);
    Mat crossingNumber2;
    threshold(input_thinned2, crossingNumber2, 0, 1, THRESH_BINARY | THRESH_OTSU);
    createCrossingNumber(crossingNumber2, harris_corners2);
    normalize(harris_corners2, harris_normalised2, 0, 255, NORM_MINMAX, CV_8UC1, Mat());
    vector<KeyPoint> keypoints2;
    Mat rescaled2;
    convertScaleAbs(harris_normalised, rescaled);
    Mat harris_c2(rescaled.rows, rescaled.cols, CV_8UC3);
    Mat in2[] = { rescaled, rescaled, rescaled };
    int from_to2[] = { 0,0, 1,1, 2,2 };
    mixChannels( in, 3, &harris_c, 1, from_to, 3 );
    for(int x=0; x<harris_corners2.cols; x++){
        for(int y=0; y<harris_corners2.rows; y++){
            if ((int)harris_corners2.at<uchar>(y,x)==3 || (int)harris_corners2.at<uchar>(y,x)==1){
                circle(harris_c2, Point(x, y), 5, Scalar(0,255,0), 1);
                circle(harris_c2, Point(x, y), 1, Scalar(0,0,255), 1);
                keypoints2.push_back( KeyPoint (x, y, 1) );
            }
        }
    }
    Mat descriptors2;
    orb_descriptor->compute(input_thinned2, keypoints2, descriptors2);

    //printf("problemeavant\n");
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    vector< DMatch > matches;
    matcher->match(descriptors, descriptors2, matches);
    //printf("%f\n",descriptors.at<float>(3,7));
    int score=0;
    if (!matches.empty()){
        for (int i=0;i<matches.size();i++){
            if (matches[i].distance<40.0){
                printf("%f\n",matches[i].distance);
                score++;
            }
        }    
    }
    printf("score : %d\n",score);
    // int score = 0;
    // for(int i=0; i < matches.size(); i++){
    //     for(int j=0;j<matches[i].size();j++){
    //         if (!matches.empty()){
    //             DMatch current_match = matches[i][j];
            
    //             printf("idTRAIN %d, idQUERRY %d, score %f\n",current_match.trainIdx,current_match.queryIdx,current_match.distance);
    //         }
    //         //printf("%f\n",current_match.distance);
    //         //if (current_match.distance>=0){score++;}
    //         //printf("%f\n",current_match.distance);
    //     }
    // }
    //printf("problemeapres\n");
    //printf("CN\t%d\t%s\t%s\t%d\t%d\t%d\t%f\n",3,nom1,nom2,(int)keypoints.size(),(int)keypoints2.size(),score,radiusSize);
    
    return(score);

}




int main( int argc, const char** argv )
{
    //compareWithCN("FP3/101_1.tif", "FP3/101_6.tif", 70);
    //for(int i=75;i<85;i+=5){
        //compareWithHaris("cleaned/101_1_cleaned.tif", "cleaned/101_2_cleaned.tif", 125, (double)i+15);
        compareWithCN("cleaned/108_4_cleaned.tif","cleaned/108_5_cleaned.tif",(double)100.0);
        
        
        
    //}
    // for(int i=10;i<120;i+=5){
    // //compareWithHaris("cleaned/101_1_cleaned.tif", "cleaned/101_2_cleaned.tif", 125, (double)i+15);
    // compareWithCN("cleaned/101_1_cleaned.tif","cleaned/104_7_cleaned.tif",(double)i);
        
        
        
    // }
    
    
    
    

    return 0;
}