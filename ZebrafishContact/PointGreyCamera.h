///////////////////////////////////////////////////////////
//
// PointGreyCamera.h
// Copyright (c) 2017 - Kim Lab.  All Rights Reserved
//
///////////////////////////////////////////////////////////

#ifndef FlyCapture2_H
#define FlyCapture2_H
#include <string>
#include <memory.h>
#include <stdint.h>
#include "FlyCapture2.h"
using namespace FlyCapture2;

struct ImgData {
    unsigned int timestamp = 0; // from the camera
    struct timeval time; // recording start time
    int32_t FrameNo = 0; // frame number / frame rate
    float x = 0;  // width
    float y = 0;  // height
    float th = 0;  //
    int32_t size = 0;
};

class PointGreyCamera
{
public:
    PointGreyCamera(void);
    ~PointGreyCamera();
	// stop and start
	volatile bool b_stop;
	volatile bool b_start;
	// -------------
	FlyCapture2::Camera cam;
    FlyCapture2::Image rawImage;
	char errMsg[1024];

	unsigned int getSerialNumber(void);
	void setSerialNumber(unsigned int _serial);

	void setROI(unsigned int _OffsetX, unsigned int _OffsetY, unsigned int _Width, unsigned int _Height);
	void setExpTime(float _ExpTime);

	bool connectCamera(void);
	bool disconnectCamera(void);
	bool releaseCamera(void);

	bool setConfigurations(void);
	bool startCapture(void);
	void resetcount(void);
	bool stopCapturing(void);
	bool grabImage(void);
    bool isCaptureEnabled(void);
    FlyCapture2::TimeStamp getTimeStamp(void);

    uint8_t * getDataPointer(void);
    unsigned int getrowBytes(void);
	unsigned int getrows(void);
	unsigned int getcols(void);
    uint64_t getFrameCount(void);
	FlyCapture2::Image getImageCopy(void);
	unsigned int ROIWidth;
	unsigned int ROIHeight;
	unsigned int ROIsz;
	unsigned int ROIsz_byte;
    float frameRate;
	double bytePerPx;
    float gain;

    uint8_t ** allImages;
    ImgData * allData;
    ImgData curData;
    uint64_t imgcntMax;
    void image_malloc(void);
    void image_mallocfree(void);
    uint64_t image_getIDSavedData(uint64_t _FrmNo);
    void cpyToSaveData(FlyCapture2::Image * _img);
    void cpyToSaveData(void);
    uint8_t * getSavedImg(uint64_t _FrmNo);
    ImgData * getSavedImgData(uint64_t _FrmNo);


	FlyCapture2::Error error;
	FlyCapture2::FC2Version fc2Version;
	FlyCapture2::PGRGuid guid;
	FlyCapture2::BusManager busMgr;
	FlyCapture2::CameraInfo NIRCameraInfo;
	FlyCapture2::TriggerMode mTriggerMode;
	FlyCapture2::Property propShutter;
	FlyCapture2::Property propGain;
	FlyCapture2::StrobeControl mStrobe;
	FlyCapture2::Format7ImageSettings mImageSetting;
	FlyCapture2::ImageMetadata metadata;
	FlyCapture2::FC2Config mCamearConfig;
	FlyCapture2::PixelFormat mPixelFormat;
    FlyCapture2::FrameRate mFrameRateMode;




	FlyCapture2::Property propGamma;
	FlyCapture2::Property propExp;
	FlyCapture2::Property propBrightness;

	unsigned int droppedFrames;
	bool setGain(float);
    float _fps;
    bool setShutterSpeed(float _exp);


private:
	unsigned int SerialNoNIRCamera;
	float ExposureTime;		
	unsigned int ROIoffsetX;
	unsigned int ROIoffsetY;
	bool CaptureEnabled;	
    uint64_t FrameCount;
    uint64_t InitFrameCount;

	void errorMsgDisp(void);
	void getCameraInfo(void);
    bool setExtTriggerMode(bool _enable);
    bool setShutterSpeed(void);
    bool setFrameRate(void);
	bool setFrameRate(float fps);
	bool setOutputLine(bool _enable, float _Pulsewidth);
	bool setEmbeddedImagePropertyData(void);
	bool setROICamera(void);
	bool setTriggerOff(void);
	bool setFC2Config(void);
    bool setGain(void);

	enum{FallingEdge, RisingEdge};
};

#endif // FlyCapture2_H
