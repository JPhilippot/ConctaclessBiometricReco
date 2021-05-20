#include "opencv4/opencv2/opencv.hpp"
#include <cstdio>
#include <opencv2/core/base.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv4/opencv2/core/cvstd_wrapper.hpp>
#include <opencv4/opencv2/core/hal/interface.h>
#include <opencv4/opencv2/xfeatures2d.hpp>
#include <fstream>
#include <bitset>
#include <string>

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

float distance(float x1, float y1, float x2, float y2){
    return std::sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
}
float distance(Point2f p1, float x2, float y2){
    return std::sqrt((p1.x-x2)*(p1.x-x2)+(p1.y-y2)*(p1.y-y2));
}
float distance(Point2f p1, Point2f p2){
    return std::sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y));
}


std::vector<KeyPoint>* pickDistantPoints(std::vector<KeyPoint>* sortedPoints, double radius, int pointsNbr){ //sorted points -> around centroid (min dist)

    //std::vector<KeyPoint> leftToAnalyse = std::vector<KeyPoint>(*sortedPoints); //useless ?
    std::vector<KeyPoint>* cherrypicked = new std::vector<KeyPoint>();

    cherrypicked->push_back((*sortedPoints)[0]);//leftToAnalyse.erase(leftToAnalyse.begin());

    int i =1; 
    while((int)cherrypicked->size()<pointsNbr && i<(int)sortedPoints->size()){
        bool flag = true;
        for (int p=0; p<(int)cherrypicked->size();p++){
            flag = flag && distance((*cherrypicked)[p].pt,(*sortedPoints)[i].pt)>radius;
            if (!flag){break;}
        }
        if (flag){
            cherrypicked->push_back((*sortedPoints)[i]);//leftToAnalyse.erase(leftToAnalyse.begin()+i);
        }
        i++;
    }
    
    return(cherrypicked);
}

std::vector<KeyPoint>* centroidPicking(std::vector<KeyPoint> &listePoints, double radius, int nombreMinutiaeFinal){


    float centroidX =0.0;                           
    float centroidY =0.0;

    for (int k=0;k<(int)listePoints.size();k++){   
        centroidX+=listePoints[k].pt.x;
        centroidY+=listePoints[k].pt.y;
    }
    centroidX /=(float)listePoints.size();
    centroidY/=(float)listePoints.size();
    

    //cherrypick centered minutiae (ugly sorting algorithm ?)

    std::vector<KeyPoint>* res = new std::vector<KeyPoint>();
    for(int c =0; c<(int)listePoints.size();c++){
        float minDist = distance(listePoints[0].pt, centroidX,centroidY);
        int minK=0;
        for (int k=0;k<listePoints.size();k++){
            if (distance(listePoints[k].pt, centroidX,centroidY)<minDist){
                minDist=distance(listePoints[k].pt, centroidX,centroidY);
                minK=k;
            }
        }
        res->push_back(listePoints[minK]);
        listePoints.erase(listePoints.begin()+minK);
    }

    std::vector<KeyPoint>* resPrime = pickDistantPoints(res, radius, nombreMinutiaeFinal);

    return (resPrime);


}





std::vector<KeyPoint>* KMeansClusterization(std::vector<KeyPoint>& minutiae, int nombreMinutiaeFinal){

    std::vector<KeyPoint>* centroidArray = new std::vector<KeyPoint>();

    std::vector<Point2f> clusterCenters = *new std::vector<Point2f>();

// on stocke ici les clusters .... useless ?
    // std::vector<std::vector<Point2f>> clusteredMinutiae = *new std::vector<std::vector<Point2f>>();
    // for(int i=0;i<nombreMinutiaeFinal;i++){
    //     clusteredMinutiae[i]= *new std::vector<Point2f>();
    // }


    int randIndex = (int)(rand()%minutiae.size());
    KeyPoint c1 = minutiae[0];
    //centroidArray->push_back(c1);

    clusterCenters.push_back(c1.pt);

    // float meanDist = 0.0;
    // for (int k=0; k<(int)minutiae.size();k++){
    //     meanDist+=distance(c1.pt,minutiae[k].pt);
    // }
    // meanDist/=(float)minutiae.size();


    //KeyPoint c2;

//comment recuperer un point "assez distant" ? peut etre prendre plus loin ...
    // for(int k=0;k<(int)minutiae.size();k++){
    //     if (k!=randIndex){
    //         if (distance(c1.pt, minutiae[k].pt)/meanDist>2.0 || distance(c1.pt, minutiae[k].pt)/meanDist<0.5){
    //             c2=minutiae[k];
    //             //printf("c'est bon j'ai trouve");
    //             break;
    //         }
    //     }
    // }
    //centroidArray->push_back(c2);

//premiere clusterisation
    // for(int k=0;k<(int)minutiae.size();k++){
    //     if (distance(minutiae[k].pt,centroidArray[0])<distance(minutiae[k].pt,centroidArray[1])){
    //         clusteredMinutiae[0].push_back(minutiae[k].pt);
    //     }
    //     else{clusteredMinutiae[1].push_back(minutiae[k].pt);}
    // }


//construction de centroids de clusters au fur et a mesure //ET remplissage tabClusters
    for (int c=1;c<nombreMinutiaeFinal;c++){

    //trouver le point le plus distant => nouveau centre de cluster
        float furthestDist = 0.0;
        int furthestPointIndice=0;
        for (int k =0;k<(int)(minutiae.size());k++){
            float dist = 0.0;
            for (int center=0;center<c;center++){
                dist+=distance(minutiae[k].pt,(clusterCenters)[center]);
            }
            if (dist>furthestDist){
                    furthestDist = dist;
                    furthestPointIndice=k;
            }
        }
    //furthest point has been found. must now add it to clusterCentroids
        clusterCenters.push_back(minutiae[furthestPointIndice].pt);

    //cluster centers become their mean value ; then repeat
        Point2f* tempCenters = new Point2f[c];
        int* tabOcc = new int[c];
        for (int k = 0; k<(int)minutiae.size();k++){
            float minDist = distance(minutiae[k].pt,clusterCenters[0]);
            int clusterNumber = 0;
            for(int cluster = 0 ; cluster<c;cluster++){
                if (distance(minutiae[k].pt,clusterCenters[cluster])<minDist){
                    minDist=distance(minutiae[k].pt,clusterCenters[cluster]);
                    clusterNumber=cluster;
                }

            }
            tempCenters[clusterNumber].x+=minutiae[k].pt.x;
            tempCenters[clusterNumber].y+=minutiae[k].pt.y;
            tabOcc[clusterNumber]++;
        }
        for (int i = 0;i<c;i++){
            tempCenters[i].x/=(float)tabOcc[i];
            tempCenters[i].y/=(float)tabOcc[i];
            clusterCenters[i]=tempCenters[i];
        }
        
    }
    //printf("%d taille\n",(int)centroidArray->size());

    // for (int k=0;k<(int)centroidArray->size();k++){
    //     printf("x : %f     y : %f\n",(*centroidArray)[k].pt.x,(*centroidArray)[k].pt.y);
    // }
    for (int i =0;i<nombreMinutiaeFinal;i++){
        centroidArray->push_back(*new KeyPoint(clusterCenters[i].x,clusterCenters[i].y,1));
    }

    return centroidArray;
    

}


void compareWithCN(string nom1, string nom2, double radiusSize, bool samePerson, int* VP, int* FP, int*VN, int* FN){


    double delta1 = 10.0;


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


    // Mat container(input.rows, input.cols*2, CV_8UC1);
    // input.copyTo( container( Rect(0, 0, input.cols, input.rows) ) );
    // input_binary.copyTo( container( Rect(input.cols, 0, input.cols, input.rows) ) );
    


    for(int x=0; x<harris_corners.cols; x++){
        for(int y=0; y<harris_corners.rows; y++){
            if ((int)harris_corners.at<uchar>(y,x)==1 || (int)harris_corners.at<uchar>(y,x)==3){
                //circle(harris_c, Point(x, y), 5, Scalar(0,255,0), 1);
                circle(harris_c, Point(x, y), 2, Scalar(0,0,255), 1);
                keypoints.push_back( KeyPoint (x, y, 1) );
            }
        }
    }
    

    //compute the barycentre 

    std::vector<KeyPoint> cherrypicked = *centroidPicking(keypoints, delta1,  40);

    //OR use kmeans clusters :

    //std::vector<KeyPoint> cherrypicked = *KMeansClusterization(keypoints, 20);
    for(int pick =0;pick<cherrypicked.size();pick++){
        circle(harris_c, Point(cherrypicked[pick].pt.x, cherrypicked[pick].pt.y), 5, Scalar(0,255,0), 1);
    }
        
    imshow("input versus binary", harris_c); waitKey(0);


    Ptr<Feature2D> orb_descriptor = ORB::create();
    Mat descriptors;


    orb_descriptor->compute(input_thinned, cherrypicked, descriptors); //cherrypicked instead



    // for (int i =0; i<descriptors.cols;i++){
    //     for (int j =0; j<descriptors.rows;j++){
    //        printf("%d %d %f \n",i,j,descriptors.at<float>(i,j));
    //     }
    // }



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
    convertScaleAbs(harris_normalised2, rescaled2);
    Mat harris_c2(rescaled2.rows, rescaled2.cols, CV_8UC3);
    Mat in2[] = { rescaled2, rescaled2, rescaled2 };
    int from_to2[] = { 0,0, 1,1, 2,2 };
    mixChannels( in2, 3, &harris_c2, 1, from_to2, 3 );
    for(int x=0; x<harris_corners2.cols; x++){
        for(int y=0; y<harris_corners2.rows; y++){
            if ((int)harris_corners2.at<uchar>(y,x)==1 || (int)harris_corners2.at<uchar>(y,x)==3){
                //circle(harris_c2, Point(x, y), 5, Scalar(0,255,0), 1);
                circle(harris_c2, Point(x, y), 2, Scalar(0,0,255), 1);
                keypoints2.push_back( KeyPoint (x, y, 1) );
            }
        }
    }


    std::vector<KeyPoint> cherrypicked2 = *centroidPicking(keypoints2,delta1, 40);
    //std::vector<KeyPoint> cherrypicked2 = *KMeansClusterization(keypoints2, 20);

    for(int pick =0;pick<cherrypicked2.size();pick++){
        circle(harris_c2, Point(cherrypicked2[pick].pt.x, cherrypicked2[pick].pt.y), 5, Scalar(0,255,0), 1);
    }
        
    imshow("input versus binary", harris_c2); waitKey(0);


    Mat descriptors2;
    orb_descriptor->compute(input_thinned2, cherrypicked2, descriptors2);

    //printf("problemeavant\n");
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    vector< DMatch > matches;

    
    //matcher->match(descriptors, descriptors2, matches);
    //printf("%f\n",descriptors.at<float>(3,7));


    
    matcher->match(descriptors,descriptors2,matches);

    if (!matches.empty()){
        for (int i=0;i<matches.size();i++){
            if (matches[i].distance<=radiusSize){
                if (samePerson){(*VP)++;}
                if (!samePerson){(*FP)++;}
            }
            else{
                if (samePerson){(*FN)++;}
                if (!samePerson){(*VN)++;}
            }
        }    
    }


    
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
    
    //printf("VP : %d FP : %d VN : %d FN : %d\n",VP,FP,VN,FN);

}




int main( int argc, const char** argv )
{
    //compareWithCN("FP3/101_1.tif", "FP3/101_6.tif", 70);

    double minDist = 100.0;
    for(int i=10;i<=100;i+=10){

        int *VP=new int; *VP=0;
        int *FP=new int; *FP=0;
        int *VN=new int; *VN=0;
        int *FN=new int; *FN=0;
        
        //for (int personne = 1;personne<10;personne++){
            //for (int image =1;image<5;image++){
                //if (!(personne==9 && image==1)){
                    // std::string deb = std::string("cleaned/10");
                    // deb += std::to_string(personne);
                    // deb += std::string("_");
                    // deb+=std::to_string(image);
                    // deb+=std::string("_cleaned.tif");
                    compareWithCN(string("cleaned/106_6_cleaned.tif"),string("cleaned/106_7_cleaned.tif"),(double)i,!true, VP, FP, VN, FN);
                    printf("radius : %d    VP : %d FP : %d VN : %d FN : %d\n",i,*VP,*FP,*VN,*FN);
                //}
            //}
        //}

        
        double sensitivite = (double)((double)*VP/(double)(*VP+*FN));
        double specificite = 1.0-(double)((double)*VN/(double)(*VN+*FP));
        //minDist=std::min(minDist,sqrt(pow(0.0-sensitivite,2.0)+pow(1.0-specificite,2.0)));
        //printf("%d %f %f %f\n",i,sensitivite,specificite,sqrt(pow(0.0-sensitivite,2.0)+pow(1.0-specificite,2.0)));
        
        
    }

    
    
    
    

    return 0;
}