#include "functions.h"
#include "binPeopleDetector.h"

int main()
{
    cv::VideoCapture capture(1);
    
    //模组摄像头分辨率必须是(640*240)的一或二或四倍
    capture.set(CV_CAP_PROP_FRAME_WIDTH,640);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT,240);
    
    cv::Mat frame,frame1,frame2;
    
    binPeopleDetector detector;
    
    while(1)
    {
        //捕捉一帧
        capture >> frame;
        
        //分割左右
        frame1 = frame(cv::Range(0,frame.rows),cv::Range(0,frame.cols/2));
        frame2 = frame(cv::Range(0,frame.rows),cv::Range(frame.cols/2,frame.cols));
        
        //水平对准(模组摄像头的deltaY=-2)
        horizonAlign(frame1,frame2,-2);
        
        //行人检测
        std::vector<cv::Rect> faceRects1;
        std::vector<cv::Rect> faceRects2;
        detector.detectFace(frame1,frame2,faceRects1,faceRects2);
        
        //test
        for(int i=0;i<faceRects1.size();i++)
        {
            cv::rectangle(frame1,faceRects1[i],cv::Scalar(255,0,0));
            cv::putText(frame1,std::to_string(i),faceRects1[i].tl(),0,1,cv::Scalar(0,255,0),1);
        }
        for(int i=0;i<faceRects2.size();i++)
        {
            cv::rectangle(frame2,faceRects2[i],cv::Scalar(255,0,0));
            cv::putText(frame2,std::to_string(i),faceRects2[i].tl(),0,1,cv::Scalar(0,255,0),1);
        }
        
        //对每一个行人进行双目测距
        //cv::Rect rect1(50,50,100,60);
        //cv::Mat mask1(frame1.size(),CV_8U);
        //mask1 = cv::Scalar(0);
        //cv::rectangle(mask1,rect1,cv::Scalar(255),-1);
        //float distance = binDistMeasure(50,frame1,mask1,frame2,mask1);
        
        //test
        cv::namedWindow("frame1",0);
        cv::namedWindow("frame2",0);
        cv::imshow("frame1",frame1);
        cv::imshow("frame2",frame2);
        
        if(cv::waitKey(1) == 'q')
            break;
    }
    
    return 0;
}
