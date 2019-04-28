#include "PointGreyCamSpinnaker.h"

PointGreyCamSpinnaker::PointGreyCamSpinnaker(void)
{

    char cmd[256];
    char password[256] = "nvidia";
    sprintf(cmd, "echo %s | sudo -S sh -c 'echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb'", password);
    system(cmd);

    unsigned int sn = 18575590; // grasshopper

	b_start = false;
	b_stop = false;



//    setSerialNumber(sn);
//    setROI(0, 0, 1140, 1080);

//    CaptureEnabled = false;
//	FrameCount = 0;
//	InitFrameCount = 0;
//	droppedFrames = 0;
//    bytePerPx = 1;
//    ExposureTime = 10.0f;
//    frameRate = 225.0f;
//    gain = 20.0f;
//    allImages = NULL;
//    imgcntMax = 200;

}

PointGreyCamSpinnaker::~PointGreyCamSpinnaker()
{
}

//void PointGreyCamera::setSerialNumber(unsigned int _serial) {
//	SerialNoNIRCamera = _serial;
//}

//unsigned int PointGreyCamera::getSerialNumber(void) {
//	return SerialNoNIRCamera;
//}

//void PointGreyCamera::setROI(unsigned int _OffsetX, unsigned int _OffsetY, unsigned int _Width, unsigned int _Height) {
//	ROIoffsetX = _OffsetX;
//	ROIoffsetY = _OffsetY;
//	ROIWidth = _Width;
//	ROIHeight = _Height;

//	ROIsz = ROIWidth*ROIHeight;
//	ROIsz_byte = ROIsz * bytePerPx;
//}

//void PointGreyCamera::setExpTime(float _ExpTime) {
//	ExposureTime = _ExpTime;
//}

//bool PointGreyCamera::connectCamera(void) {
//	error = busMgr.GetCameraFromSerialNumber(SerialNoNIRCamera, &guid);
//    if (error.GetType() != PGRERROR_OK) {
//        errorMsgDisp(); return false;
//    }
//    error = cam.Connect(&guid);
//    if (error.GetType() != PGRERROR_OK) {
//        errorMsgDisp(); return false;
//    }
//	return true;
//}
	

//bool PointGreyCamera::setConfigurations(void) {
//    mFrameRateMode = FRAMERATE_FORMAT7;

//    if (!setFrameRate()) return false;
//    if (!setExtTriggerMode(false)) return false;
//    if (!setShutterSpeed()) return false;
//    if (!setOutputLine(true, ExposureTime)) return false;
//    if (!setEmbeddedImagePropertyData()) return false;
//    if (!setROICamera()) return false;
//    if (!setFC2Config()) return false;
//    if (!setGain()) return false;
//	return true;
//}

//bool PointGreyCamera::startCapture(void) {
//    error = cam.StartCapture();
//    if (error.GetType() != PGRERROR_OK) {
//        errorMsgDisp(); return false;
//    }
//	CaptureEnabled = true;
//	FrameCount = 0;
//	InitFrameCount = 0;
//	droppedFrames = 0;
//	return CaptureEnabled;
//}

//void PointGreyCamera::resetcount(void) {
//	FrameCount = 0;
//	InitFrameCount = 0;
//	droppedFrames = 0;
//}

//bool PointGreyCamera::grabImage(void) {
//	if (CaptureEnabled) {
//		error = cam.RetrieveBuffer( &rawImage );
//        if (error.GetType() == PGRERROR_OK) {
//			metadata = rawImage.GetMetadata();
//			if  (InitFrameCount == 0)
//                InitFrameCount = (uint64_t)metadata.embeddedFrameCounter;
//            curData.timestamp = metadata.embeddedTimeStamp;
//			unsigned int FrameCount_New;
//			if (InitFrameCount <= metadata.embeddedFrameCounter)
//                FrameCount_New = (uint64_t)metadata.embeddedFrameCounter - InitFrameCount;
//			else {
//				//FrameCount_New = metadata.embeddedFrameCounter - InitFrameCount;
//                FrameCount_New = (uint64_t)metadata.embeddedFrameCounter + (UINT64_MAX - InitFrameCount);
//			}

//			if (!(FrameCount_New - FrameCount == 1 || FrameCount_New == 0))
//				droppedFrames = 0;
//			FrameCount= FrameCount_New;
//		}
//		else {
//            if (error.GetType() != PGRERROR_OK) {errorMsgDisp(); return false;}
//		}
//	}
//	else
//		return false;
//	return true;
//}
//bool PointGreyCamera::isCaptureEnabled(void) {
//    return CaptureEnabled;
//}

//FlyCapture2::TimeStamp PointGreyCamera::getTimeStamp(void) {
//    //cam.GetCycleTime()
//    FlyCapture2::TimeStamp _out;
//    cam.GetCycleTime(&_out);
//    return _out;
//}

//void PointGreyCamera::errorMsgDisp(void) {
////	error.PrintErrorTrace();
//	sprintf(errMsg,"%s",error.GetDescription());
//}

//void PointGreyCamera::getCameraInfo(void) {
//	FlyCapture2::Utilities::GetLibraryVersion( &fc2Version );
//}

//bool PointGreyCamera::setExtTriggerMode(bool _enable) {
//	// set GPIO1 : input,
//	error = cam.SetGPIOPinDirection(0, 0); // Pin to get the direction for. | 0 for input, 1 for output
//    mTriggerMode.mode = MODE_0; // mode 0
//	mTriggerMode.polarity= 0; // 0:Falling Edge, 1:Rising Edge
//	mTriggerMode.source = 0; // GPIO0
//	mTriggerMode.parameter = 0; // no parameter need
//	mTriggerMode.onOff = _enable;
//	error = cam.SetTriggerMode(&mTriggerMode);
//	return true;
//}

//bool PointGreyCamera::setGain(float _gaindB) {
//    gain = _gaindB;
//    setGain();
//}

//bool PointGreyCamera::setGain(void) {
//	// 4-3. set shutter time
//	//Define the property to adjust.
//	propGain.type = FlyCapture2::GAIN;
//	//Ensure the property is on.
//	propGain.autoManualMode = false; //Ensure auto-adjust mode is off.
//	propGain.absControl = true; //Ensure the property is set up to use absolute value control.
//    propGain.absValue = gain;
//	error = cam.SetProperty(&propGain);
//    if (error.GetType() != PGRERROR_OK) { errorMsgDisp(); return false; }
//	return true;
//}


//bool PointGreyCamera::setShutterSpeed(float _exp) {
//    ExposureTime = _exp;
//    setShutterSpeed();
//}

//bool PointGreyCamera::setShutterSpeed(void) {
//	// 4-3. set shutter time
//	//Define the property to adjust.
//	propShutter.type = FlyCapture2::SHUTTER;
//	//Ensure the property is on.
//	propShutter.onOff = true;
//	propShutter.autoManualMode = false; //Ensure auto-adjust mode is off.
//	propShutter.absControl = true; //Ensure the property is set up to use absolute value control.
//    propShutter.absValue = ExposureTime;//1.0; //Set the absolute value of shutter to 20 ms.
//	error = cam.SetProperty( &propShutter );
//    if (error.GetType() != PGRERROR_OK) {errorMsgDisp(); return false;}

//    propExp.type = FlyCapture2::AUTO_EXPOSURE;
//    propExp.onOff = false;
//    propExp.autoManualMode = false; //Ensure auto-adjust mode is off.
//    propExp.absControl = true; //Ensure the property is set up to use absolute value control.
//    propExp.absValue = -7.585;//;
//    error = cam.SetProperty(&propExp);

//    if (error.GetType() != PGRERROR_OK) { errorMsgDisp(); return false; }

//	return true;
//}

//bool PointGreyCamera::setFrameRate(float fps) {
//    frameRate = fps;
//    setFrameRate();
//}

//bool PointGreyCamera::setFrameRate(void) {
//	FlyCapture2::Property propFrameRate;
//    propFrameRate.type = FlyCapture2::FRAME_RATE;
//	propFrameRate.onOff = true;
//	propFrameRate.autoManualMode = false; //Ensure auto-adjust mode is off.
//	propFrameRate.absControl = true; //Ensure the property is set up to use absolute value control.
//    propFrameRate.absValue = frameRate; //Set the absolute value of shutter to 20 ms.
//	error = cam.SetProperty( &propFrameRate );
//    if (error.GetType() != PGRERROR_OK) {errorMsgDisp(); return false;}
//	return true;
//}

//bool PointGreyCamera::setOutputLine(bool _enable, float _Pulsewidth) {
//	// set output (maybe replace to the exposureActive later)
//	mStrobe.onOff = _enable;
//    mStrobe.source = 2;// GPIO1
//    mStrobe.polarity = 1; // set Low (Activate LED: on-time)
//	mStrobe.delay = 0.0f;
//	mStrobe.duration = _Pulsewidth;
//	error = cam.SetStrobe(&mStrobe);
//    if (error.GetType() != PGRERROR_OK) {errorMsgDisp(); return false;}

//    mStrobe.source = 3;// GPIO1
//    mStrobe.polarity = 1; // set Low (Activate LED: on-time)
//    mStrobe.delay = 0.0f;
//    mStrobe.duration = _Pulsewidth;
//    error = cam.SetStrobe(&mStrobe);
//    if (error.GetType() != PGRERROR_OK) {errorMsgDisp(); return false;}
//	return true;
//}
//bool PointGreyCamera::setEmbeddedImagePropertyData(void) {
//	// output image information setting
//	FlyCapture2::EmbeddedImageInfoProperty OnProperty; OnProperty.onOff = true; OnProperty.available = true;
//	FlyCapture2::EmbeddedImageInfoProperty DisabledProperty; DisabledProperty.onOff = false; DisabledProperty.available = false;
//	FlyCapture2::EmbeddedImageInfo mEmbeddedImageInfo;
//	mEmbeddedImageInfo.brightness = DisabledProperty;
//	mEmbeddedImageInfo.exposure = DisabledProperty;
//	mEmbeddedImageInfo.gain = DisabledProperty;
//	mEmbeddedImageInfo.GPIOPinState= DisabledProperty;
//	mEmbeddedImageInfo.shutter = DisabledProperty;
//	mEmbeddedImageInfo.strobePattern= DisabledProperty;
//	mEmbeddedImageInfo.whiteBalance = DisabledProperty;
//	mEmbeddedImageInfo.ROIPosition = DisabledProperty;
//	mEmbeddedImageInfo.frameCounter = OnProperty;
//	mEmbeddedImageInfo.timestamp = OnProperty;
//	error = cam.SetEmbeddedImageInfo(&mEmbeddedImageInfo);
//    if (error.GetType() != PGRERROR_OK) {errorMsgDisp(); return false;}
//	return true;
//}
//bool PointGreyCamera::setROICamera(void) {
//    mPixelFormat = PIXEL_FORMAT_RAW8;
//	cam.SetVideoModeAndFrameRate(FlyCapture2::VIDEOMODE_FORMAT7, mFrameRateMode);
//    mImageSetting.mode = MODE_0;
//	mImageSetting.offsetX = ROIoffsetX;
//	mImageSetting.offsetY = ROIoffsetY;
//	mImageSetting.width= ROIWidth;
//	mImageSetting.height = ROIHeight;
//	mImageSetting.pixelFormat = mPixelFormat;
//	error = cam.SetFormat7Configuration(&mImageSetting, 100.0f); // 100% packet size
//    if (error.GetType() != PGRERROR_OK) {errorMsgDisp(); return false;}
//	return true;
//}


//bool PointGreyCamera::setFC2Config(void) {
//	cam.GetConfiguration(&mCamearConfig);
//	//mCamearConfig.registerTimeout = 10;DROP_FRAMES
//	mCamearConfig.numBuffers = 150;
//    mCamearConfig.grabMode = BUFFER_FRAMES;// FlyCapture2::GrabMode::BUFFER_FRAMES/DROP_FRAMES
//	mCamearConfig.highPerformanceRetrieveBuffer = true;
//	mCamearConfig.grabTimeout = 2000;
//	error = cam.SetConfiguration(&mCamearConfig);
//    if (error.GetType() != PGRERROR_OK) {errorMsgDisp(); return false;}
//	return true;
//}


////bool PointGreyCamera::releaseCamera(void) {
////	if (cam.IsConnected()) {
////		if (!setTriggerOff()) return false;
////		if (!stopCapturing()) return false;
////		if (!disconnectCamera()) return false;
////	}
////	return true;
////    //cam.GetStats()
////}

////Pikam Edit Trial Start

//bool PointGreyCamera::releaseCamera(void) {
//    if (cam.IsConnected()) {
//        if (!setTriggerOff()) return false;
//        if (!stopCapturing()) return false;
//        if (!disconnectCamera()) return false;
//    }
//    return true;
//}


//bool PointGreyCamera::setTriggerOff(void) {
//	error = cam.GetTriggerMode( &mTriggerMode );
//	mTriggerMode.onOff = false;
//	error = cam.SetTriggerMode( &mTriggerMode );
//    if (error.GetType() != PGRERROR_OK) {errorMsgDisp(); return false;}
//	CaptureEnabled = false;
//	return true;
//}

//bool PointGreyCamera::stopCapturing(void) {
//	error = cam.StopCapture();
//    if (error.GetType() != PGRERROR_OK) {errorMsgDisp(); return false;}
//	return true;
//}

//bool PointGreyCamera::disconnectCamera(void) {
//	error = cam.Disconnect();
//    if (error.GetType() != PGRERROR_OK) {errorMsgDisp(); return false;}
//	return true;
//}


//uint8_t * PointGreyCamera::getDataPointer(void) {
//    return (uint8_t *)rawImage.GetData();
//}

//unsigned int PointGreyCamera::getrowBytes(void) {
//	return (double)rawImage.GetReceivedDataSize()/(double)rawImage.GetRows();
//}

//unsigned int PointGreyCamera::getrows(void) {
//    return ROIHeight;
//}

//unsigned int PointGreyCamera::getcols(void) {
//    return ROIWidth;
//}

//uint64_t PointGreyCamera::getFrameCount(void) {
//	return FrameCount;
//}


//FlyCapture2::Image PointGreyCamera::getImageCopy(void) {
//	FlyCapture2::Image imgCopy;
//	error = rawImage.DeepCopy(&imgCopy);
//	return imgCopy;
//}

//void PointGreyCamera::image_malloc(void) {
//    allImages = (uint8_t **)malloc(imgcntMax * sizeof(uint8_t *));
//    for (uint64_t i=0; i< imgcntMax; i++)
//        allImages[i] = (uint8_t *)malloc(ROIsz_byte * sizeof(uint8_t));
//    allData = (ImgData *)malloc(imgcntMax * sizeof(ImgData));
//}

//void PointGreyCamera::image_mallocfree(void) {
//    if (allImages) {
//        for (uint64_t i=0; i< imgcntMax; i++)
//            free(allImages[i]);
//        free(allImages);
//        free(allData);
//    }
//}
//uint64_t PointGreyCamera::image_getIDSavedData(uint64_t _FrmNo) {
//    return _FrmNo%imgcntMax;
//}

//void PointGreyCamera::cpyToSaveData(FlyCapture2::Image * _img) {
//    uint64_t id = image_getIDSavedData(FrameCount);
//    memcpy(allImages[id], _img->GetData(), ROIsz_byte);
//    allData[id] = curData;
//}

//void PointGreyCamera::cpyToSaveData(void) {
//    cpyToSaveData(&rawImage);
//}

//uint8_t * PointGreyCamera::getSavedImg(uint64_t _FrmNo) {
//    return allImages[image_getIDSavedData(FrameCount)];
//}

//ImgData * PointGreyCamera::getSavedImgData(uint64_t _FrmNo) {
//    uint64_t id = image_getIDSavedData(_FrmNo);
//    return &(allData[id]);
//}
