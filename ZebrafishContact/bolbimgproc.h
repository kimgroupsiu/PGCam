#ifndef BOLBIMGPROC_H
#define BOLBIMGPROC_H

// HEADER FILES FOR OpenCV
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;
using namespace std;

class bolbimgproc
{
public:
    bolbimgproc();
    ~bolbimgproc();
    Mat * im;
    void copyIm(char * );
    void proc(void);
    void setImSize(size_t w, size_t h, size_t _byteperpx);
    void setTheshold(double _t); // 0-1;
    void setMinObjSz(size_t _sz); // px
    Point3f getCentroid(void);
    float x(void);
    float y(void);
    float th(void);
    size_t ObjSz(void);
    Mat * BW(void);
    Point3f findMaxObjCentroid(void);
    void setEnable(bool);
    bool isEnable(void);
private:
    Mat im_bw;
    double bw_threshold;
    size_t width;
    size_t height;
    size_t byteperpx;
    size_t sz_byte;
    size_t objSz_min;
    Point3f centroid;
    size_t centroid_size;
    bool enable;
};

#endif // BOLBIMGPROC_H
