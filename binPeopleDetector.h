#ifndef BINPEOPLEDETECTOR_H
#define BINPEOPLEDETECTOR_H

#endif // BINPEOPLEDETECTOR_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

class binPeopleDetector
{
private:
    cv::HOGDescriptor m_peopleHog;
    cv::CascadeClassifier m_faceDetector;
    cv::CascadeClassifier m_upperBodyDetector;
    
    double m_hitThreshold;
    cv::Size m_winStride;
    cv::Size m_padding;
    double m_scale;
    double m_groupThreshold;
    
public:
    binPeopleDetector()
    {
        m_peopleHog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
            
        m_hitThreshold = 0;
        m_winStride = cv::Size(4,4);
        m_padding = cv::Size(32,32);
        m_scale = 1.1;
        m_groupThreshold = 10;
        
        m_faceDetector.load("../PeopleDetection/xmls/haarcascade_frontalface_default.xml");
        m_upperBodyDetector.load("../PeopleDetection/xmls/haarcascade_upperbody.xml");
    }
    
    void detectPeople(const cv::Mat &frame1, const cv::Mat &frame2, 
                std::vector<cv::Rect> &peopleRects1, std::vector<cv::Rect> &peopleRects2)
    {
        m_peopleHog.detectMultiScale(frame1,peopleRects1,m_hitThreshold,m_winStride,m_padding,m_scale,m_groupThreshold);
        m_peopleHog.detectMultiScale(frame2,peopleRects2,m_hitThreshold,m_winStride,m_padding,m_scale,m_groupThreshold);
    }
    
    void detectFace(const cv::Mat &frame1, const cv::Mat &frame2, 
                    std::vector<cv::Rect> &faceRects1, std::vector<cv::Rect> &faceRects2)
    {
        m_faceDetector.detectMultiScale(frame1,faceRects1);
        m_faceDetector.detectMultiScale(frame2,faceRects2);
    }
    
    void detectUpperBody(const cv::Mat &frame1, const cv::Mat &frame2, 
                    std::vector<cv::Rect> &ubRects1, std::vector<cv::Rect> &ubRects2)
    {
        m_upperBodyDetector.detectMultiScale(frame1,ubRects1);
        m_upperBodyDetector.detectMultiScale(frame2,ubRects2);
    }

};
