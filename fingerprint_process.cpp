#include "opencv4/opencv2/opencv.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv4/opencv2/xfeatures2d.hpp>
#include <fstream>

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

int main( int argc, const char** argv )
{
    // Read in an input image - directly in grayscale CV_8UC1
    // This will be our test fingerprint
    Mat input = imread("FP3/101_1.tif", IMREAD_GRAYSCALE);
    if(input.empty()){
	    cerr << "Image not read correctly. Check if path is correct!" << endl;
    }

    // Binarize the image, through local thresholding
    Mat input_binary;
    threshold(input, input_binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

    // Compare both
    Mat container(input.rows, input.cols*2, CV_8UC1);
    input.copyTo( container( Rect(0, 0, input.cols, input.rows) ) );
    input_binary.copyTo( container( Rect(input.cols, 0, input.cols, input.rows) ) );
    imshow("input versus binary", container); waitKey(0);

    // Now apply the thinning algorithm
    Mat input_thinned = input_binary.clone();
    thinning(input_thinned);

    // Compare both
    input_binary.copyTo( container( Rect(0, 0, input.cols, input.rows) ) );
    input_thinned.copyTo( container( Rect(input.cols, 0, input.cols, input.rows) ) );
    imshow("binary versus thinned", container); waitKey(0);

    // Now lets detect the strong minutiae using Haris corner detection
    Mat harris_corners, harris_normalised;
    harris_corners = Mat::zeros(input_thinned.size(), CV_8UC1);


    //cornerHarris(input_thinned, harris_corners, 2, 3, 0.04, BORDER_DEFAULT);

    //normalize(input_thinned, input_thinned, 0, 1, NORM_MINMAX, CV_8UC1, Mat());
    


    input_thinned.copyTo( container( Rect(0, 0, input.cols, input.rows) ) );
    harris_corners.copyTo(container(Rect(input.cols,0,input.cols,input.rows)));

    imshow("thin vs CN", container) ; waitKey(0);


    Mat crossingNumber;

    threshold(input_thinned, crossingNumber, 0, 1, THRESH_BINARY | THRESH_OTSU);

    createCrossingNumber(crossingNumber, harris_corners);

    vector<KeyPoint> keypoints;

    for(int x=0; x<harris_corners.cols; x++){
        for(int y=0; y<harris_corners.rows; y++){
            if ((int)harris_corners.at<int>(y,x)==1 || (int)harris_corners.at<int>(y,x)==3){
                circle(harris_corners, Point(x, y), 5, Scalar(0,255,0), 1);
                circle(harris_corners, Point(x, y), 1, Scalar(0,0,255), 1);
                keypoints.push_back( KeyPoint (x, y, 1) );
            }
        }
    }


    //Sobel(input_thinned, harris_corners, 0, 2, 2);
    //normalize(harris_corners, harris_normalised, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

    // Select the strongest corners that you want
    int threshold_harris = 125;
    

    // Make a color clone for visualisation purposes
    Mat rescaled;

    //harris_normalised.copyTo(container (Rect(0,0,input.cols, input.rows)));
    input_thinned.copyTo( container( Rect(0, 0, input.cols, input.rows) ) );
    harris_corners.copyTo(container(Rect(input.cols,0,input.cols,input.rows)));

    imshow("thin vs CN 1 and 3", container) ; waitKey(0);

    // convertScaleAbs(harris_normalised, rescaled);
    // Mat harris_c(rescaled.rows, rescaled.cols, CV_8UC3);
    // Mat in[] = { rescaled, rescaled, rescaled };
    // int from_to[] = { 0,0, 1,1, 2,2 };
    // mixChannels( in, 3, &harris_c, 1, from_to, 3 );
    // for(int x=0; x<harris_normalised.cols; x++){
    //    for(int y=0; y<harris_normalised.rows; y++){
    //       //if ( (int)harris_normalised.at<float>(y, x) > threshold_harris ){
    //          // Draw or store the keypoint location here, just like you decide. In our case we will store the location of the keypoint
    //          circle(harris_c, Point(x, y), 5, Scalar(0,255,0), 1);
    //          circle(harris_c, Point(x, y), 1, Scalar(0,0,255), 1);
    //          keypoints.push_back( KeyPoint (x, y, 1) );
    //       }
    //    }
    // }
    // imshow("temp", harris_c); waitKey(0);

    // Compare both
    Mat ccontainer(input.rows, input.cols*2, CV_8UC3);
    Mat input_thinned_c = input_thinned.clone();
    cvtColor(input_thinned_c, input_thinned_c, COLOR_GRAY2BGR);
    input_thinned_c.copyTo( ccontainer( Rect(0, 0, input.cols, input.rows) ) );
    //harris_c.copyTo( ccontainer( Rect(input.cols, 0, input.cols, input.rows) ) );
    imshow("thinned versus selected corners", ccontainer); waitKey(0);

    // Calculate the ORB descriptor based on the keypoint
    Ptr<Feature2D> orb_descriptor = ORB::create();
    Mat descriptors;
    orb_descriptor->compute(input_thinned, keypoints, descriptors);
    //cout<<descriptors<<endl;
    // You can now store the descriptor in a matrix and calculate all for each image.
    // Since we just got the hamming distance brute force matching left, we will take another image and calculate the descriptors also.
    // Removed as much overburden comments as you can find them above
    Mat input2 = imread("FP3/107_5.tif", IMREAD_GRAYSCALE);
    Mat input_binary2;
    threshold(input2, input_binary2, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
    Mat input_thinned2 = input_binary2.clone();
    thinning(input_thinned2);
    Mat harris_corners2, harris_normalised2;
    harris_corners2 = Mat::zeros(input_thinned2.size(), CV_32FC1);
    cornerHarris(input_thinned2, harris_corners2, 2, 3, 0.04, BORDER_DEFAULT);
    //Sobel(input_thinned2, harris_corners2, 0, 2, 2);
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
          if ( (int)harris_normalised2.at<float>(y, x) > threshold_harris ){
             // Draw or store the keypoint location here, just like you decide. In our case we will store the location of the keypoint
             circle(harris_c2, Point(x, y), 5, Scalar(0,255,0), 1);
             circle(harris_c2, Point(x, y), 1, Scalar(0,0,255), 1);
             keypoints2.push_back( KeyPoint (x, y, 1) );
          }
       }
    }
    imshow("temp2", harris_c2); waitKey(0);
    Mat descriptors2;
    orb_descriptor->compute(input_thinned2, keypoints2, descriptors2);

    // Now lets match both descriptors
    // Create the matcher interface
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    vector< DMatch > matches;
    matcher->match(descriptors, descriptors2, matches);

    // Loop over matches and multiply
    // Return the matching certainty score
    float score = 0.0;
    for(int i=0; i < matches.size(); i++){
        DMatch current_match = matches[i];
        score = score + current_match.distance/matches.size();
    }
    cerr << endl << "Current matching score = " << score << endl;

    return 0;
}