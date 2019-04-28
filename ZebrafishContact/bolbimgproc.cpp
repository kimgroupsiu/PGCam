#include "bolbimgproc.h"



bolbimgproc::bolbimgproc()
{    
    im = NULL;
    setImSize(0,0,0);
    setTheshold(0.5);
    setMinObjSz(100.0);
    enable = false;
}

bolbimgproc::~bolbimgproc()
{
    if (!im) delete im;
}

void bolbimgproc::setEnable(bool _v) {
    enable = _v;
}

bool bolbimgproc::isEnable(void) {
    return enable;
}

void bolbimgproc::copyIm(char * _pt) {

    if (enable) {
        if (!im) {
            delete im;
            im = new cv::Mat(height, width, CV_8UC1, _pt);
        }
    }
}

void bolbimgproc::setImSize(size_t w, size_t h, size_t _byteperpx) {
    width = w;
    height = h;
    byteperpx = _byteperpx;
    sz_byte = w*h*_byteperpx;
}
void bolbimgproc::setMinObjSz(size_t _sz) {
    objSz_min = _sz;
}

void bolbimgproc::setTheshold(double _t) {
    // _t = 0 ~ 1
    if (_t>1.0) _t = 1.0;
    if (_t<0.0) _t = 0.0;
    double th_max = (255.0);
    bw_threshold = _t*th_max;
}
Point3f bolbimgproc::getCentroid(void) {return centroid;}
float bolbimgproc::x(void){return centroid.x;}
float bolbimgproc::y(void){return centroid.y;}
float bolbimgproc::th(void) {return centroid.z;}
size_t bolbimgproc::ObjSz(void) {return centroid_size;}


// -----------------------------------------------------------------
void bolbimgproc::proc(void) {
    // Binarization
    if (enable) {
        BW();
        findMaxObjCentroid();
    }
    else
    {
        centroid.x = 0;
        centroid.y = 0;
        centroid.z = 0;
        centroid_size = 0;
    }
}

Mat * bolbimgproc::BW(void){
    threshold(*im, im_bw, bw_threshold, 255.0, 0);
    return &im_bw;
}

Point3f bolbimgproc::findMaxObjCentroid(void) {
    // Find maximum contour
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    int savedContour = -1; // contour index (largest contour)
    double maxArea = 0.0; // size of the largest contour
    findContours(im_bw, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0,0));
    for (int ii = 0; ii < contours.size(); ii++)
    {
        double area = contourArea(contours[ii]);
        if (area > maxArea)
        {
            maxArea = area;
            savedContour = ii;
        }
    }
    centroid_size = maxArea;
    // COMPTUTING THE CENTROID USING THE MOMENTS
    double _cx = 0;
    double _cy = 0;
    double _th = 0;
    if (maxArea > 100.0) {
      Moments mu = moments(contours[savedContour], false );
      // compute centroid & orientation
      _cx = mu.m10/mu.m00;
      _cy = mu.m01/mu.m00;
      double mu20p = mu.m20/mu.m00-_cx*_cx;
      double mu02p= mu.m02/mu.m00-_cy*_cy;
      double mu11p= mu.m11/mu.m00-_cx*_cy;
      _th = atan2(2*mu11p,(mu20p-mu02p))/2; // in radian
    }
    centroid.x = (float)_cx;
    centroid.y = (float)_cy;
    centroid.z = _th;
    return centroid;
}

