#include "camera_spk.h"

camera_spk::camera_spk()
{
    setSerialNo("18575590"); // camera serial number (GS3)
    setPassword("nvidia"); // set the sudo password
    setJetsonTX2_usbbandwidth(); // run for Jetson tx2 only. otherwise do not run;

    sys = NULL;
    numCameras = 0;
    pCam = NULL;
    isReady = false;
    frmRate_max = 0;
    frmRate = 0;

    bytePerPx = 1;
    allImages = NULL;
    testIm = NULL;
    imgcntMax = 500;
    ROIsz_byte = 0;
    FrameCount = 0;
}

camera_spk::~camera_spk() {
    isReady = false;
    if (allImages) {
        for (uint64_t i=0; i< imgcntMax; i++)
            free(allImages[i]);
        free(allImages);
        free(testIm);
    }
    closeCamera();
}

void camera_spk::setJetsonTX2_usbbandwidth(void) {
    // To acquire images greater than 2 MB in resolution
    char cmd[256];
    sprintf(cmd, "echo %s | sudo -S sh -c 'echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb'", password);
    system(cmd);
    // sudo sh -c 'echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb' in terminal
}


bool camera_spk::openCamera(void) {
    loadCamlist_Spinnaker(); // get a list of the camera -> saved to numCameras
    if (numCameras == 0) // if there is no camera, return false
        return false;
    try {
        pCam = camList.GetBySerial(serial); // get a camera with pre-saved serial number
    }
    catch (Spinnaker::Exception &e)
    {
        cout << "Error: " << e.what() << endl;
        return false;
    }

    return true;
}

unsigned int camera_spk::loadCamlist_Spinnaker(void) {
    // return number of devices connected
    // get instance
    sys = System::GetInstance();
    // Retrieve list of cameras from the system
    camList = sys->GetCameras();
    numCameras = camList.GetSize();
    return numCameras;
}

void camera_spk::closeCamera(void) {

    try {
        stopAcquisition();
        // Deinitialize camera
        pCam->DeInit();
        if (numCameras > 0) {
            // Release reference to the camera
            pCam = NULL;
            // Clear camera list before releasing system
            camList.Clear();
            // Release system
            sys->ReleaseInstance();
        }
    }
    catch (Spinnaker::Exception &e)
    {
        cout << "Error: " << e.what() << endl;
    }

}

int camera_spk::beginAcquisition(void)
{
    int result = 0;
    try
    {
        FrameCount = 0;
        pCam->BeginAcquisition();
    }
    catch (Spinnaker::Exception &e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

int camera_spk::stopAcquisition(void) {
    int result = 0;
    try
    {
        // *** NOTES ***
        // Images retrieved directly from the camera (i.e. non-converted
        // images) need to be released in order to keep from filling the
        // buffer.
        //
        pResultImage->Release();
        pCam->EndAcquisition();
    }
    catch (Spinnaker::Exception &e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

void camera_spk::setSerialNo(char * _no) {
    sprintf(serial, "%s", _no);
}

void camera_spk::setPassword(char * _no) {
    sprintf(password, "%s", _no);
}


int camera_spk::initCamera(void)
{
    int result = 0;
    try
    {
        pCam->Init();
    }
    catch (Spinnaker::Exception &e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}


bool camera_spk::grabImage(void) {
    bool result = false;
    try
    {
        // Retrieve next received image
        pResultImage = pCam->GetNextImage();
        // *** NOTES ***
        // Images can easily be checked for completion. This should be
        // done whenever a complete image is expected or required.
        // Further, check image status for a little more insight into
        // why an image is incomplete.
        //
        if (pResultImage->IsIncomplete())
        {
            // Retreive and print the image status description
            cout << "Image incomplete: "
                << Image::GetImageStatusDescription(pResultImage->GetImageStatus())
                << "..." << endl << endl;
        }
        else
        {
            isReady = true;
            result = true;
            FrameCount++;
            //
            // Print image information; height and width recorded in pixels
            //
            // *** NOTES ***
            // Images have quite a bit of available metadata including
            // things such as CRC, image status, and offset values, to
            // name a few.
            //
/*
            // cout << "Grabbed image " << imageCnt << ", width = " << width << ", height = " << height << endl;

            //
            // Convert image to mono 8
            //
            // *** NOTES ***
            // Images can be converted between pixel formats by using
            // the appropriate enumeration value. Unlike the original
            // image, the converted one does not need to be released as
            // it does not affect the camera buffer.
            //
            // When converting images, color processing algorithm is an
            // optional parameter.
            //
            ImagePtr convertedImage = pResultImage->Convert(PixelFormat_Mono8, HQ_LINEAR);

            // Create a unique filename
            ostringstream filename;

            filename << "Acquisition-";
            if (deviceSerialNumber != "")
            {
                    filename << deviceSerialNumber.c_str() << "-";
            }
            filename << imageCnt << ".jpg";

            //
            // Save image
            //
            // *** NOTES ***
            // The standard practice of the examples is to use device
            // serial numbers to keep images of one device from
            // overwriting those of another.
            //
            convertedImage->Save(filename.str().c_str());

            cout << "Image saved at " << filename.str() << endl;
*/
        }
    }
    catch (Spinnaker::Exception &e)
    {
        cout << "Error: " << e.what() << endl;
        result = false;
    }
    return result;
}


// functions for Setting parameters
bool camera_spk::confset(void){
    double _exptime = 1000;
    int64_t _bufszfrm = 200;
    double _gain = 21.0;
    double _gamma = 1.25;

    if (!confset_ROI(800, 600, 0, 0))
        return false;
    if (!confset_exptime(_exptime))
        return false;
    if (!confset_gain(_gain))
        return false;
    if (!confset_gamma(_gamma))
        return false;
    if (!confset_imgBufSz(_bufszfrm))
        return false;
    if (!confset_frmRate(200))
        return false;

    if (!confset_xRev(false))
        return false;

    return true;
}



bool camera_spk::confset_exptime(double _tims_us){
    bool _result = true;
    try
    {
        // auto off --> timed mode --> set exp. time in us
        // Get node (prep)
        INodeMap & nodeMap = pCam->GetNodeMap();
        if (pCam->ExposureAuto.GetValue() != ExposureAuto_Off) {
            // Turn off exposure: ExposureAutoEnums { ExposureAuto_Off, ExposureAuto_Once, ExposureAuto_Continuous, NUM_EXPOSUREAUTO }
            CEnumerationPtr exposureAuto = nodeMap.GetNode("ExposureAuto");
            exposureAuto->SetIntValue(exposureAuto->GetEntryByName("Off")->GetValue());
            // Set exposure mode to "Timed": ExposureModeEnums {Timed, TriggerWidth, NUM_EXPOSUREMODE}
            CEnumerationPtr exposureMode = nodeMap.GetNode("ExposureMode");
            exposureMode->SetIntValue(exposureMode->GetEntryByName("Timed")->GetValue());
        }
        // set abs value of exp. time in microsecond
        CFloatPtr exposureTime = nodeMap.GetNode("ExposureTime");
        double exptime_us_max = exposureTime->GetMax();
        if (_tims_us > exptime_us_max) _tims_us = exptime_us_max;
        exposureTime->SetValue(_tims_us);
        exptime_us = pCam->ExposureTime.GetValue();
    }
    catch (Spinnaker::Exception &e)
    {
        cout << "Error: " << e.what() << endl;
        _result = false;
    }
    return _result;
}

bool camera_spk::confset_gain(double _gain){
    bool _result = true;
    try
    {
        INodeMap & nodeMap = pCam->GetNodeMap();
        if (pCam->GainAuto.GetValue() != GainAuto_Off); {
            // Turn off the Auto Gain
            CEnumerationPtr GainAuto = nodeMap.GetNode("GainAuto");
            GainAuto->SetIntValue(GainAuto->GetEntryByName("Off")->GetValue());
        }
        // Set gain
        CFloatPtr gainVaule = nodeMap.GetNode("Gain");
        gain_max = pCam->Gain.GetMax();
        if (_gain > gain_max) _gain = gain_max;
        gainVaule->SetValue(_gain);
        gain = pCam->Gain.GetValue();
    }
    catch (Spinnaker::Exception &e)
    {
        cout << "Error: " << e.what() << endl;
        _result = false;
    }
    return _result;
}

bool camera_spk::confset_gamma(double _gamma){
    bool _result = true;
    try
    {
        double a = pCam->Gamma.GetValue();
        INodeMap & nodeMap = pCam->GetNodeMap();
        // Set gamma
        CFloatPtr GammaVaule = nodeMap.GetNode("Gamma");
        gamma_max = pCam->Gamma.GetMax();
        if (_gamma > gamma_max) _gamma = gamma_max;
        GammaVaule->SetValue(_gamma);
        gamma = pCam->Gamma.GetValue();
    }
    catch (Spinnaker::Exception &e)
    {
        cout << "Error: " << e.what() << endl;
        _result = false;
    }
    return _result;
}

bool camera_spk::confset_xRev(bool _enable) {
    bool _result = true;
    try
    {
        CBooleanPtr(pCam->GetNodeMap().GetNode("ReverseX"))->SetValue(_enable);
        // bool a = pCam->ReverseX.GetValue(true);
    }
    catch (Spinnaker::Exception &e)
    {
        cout << "Error: " << e.what() << endl;
        _result = false;
    }
    return _result;
}

bool camera_spk::confset_yRev(bool _enable) {
    bool _result = true;
    try
    {
        CBooleanPtr(pCam->GetNodeMap().GetNode("ReverseY"))->SetValue(_enable);
        // bool a = pCam->ReverseY.GetValue(true);
    }
    catch (Spinnaker::Exception &e)
    {
        cout << "Error: " << e.what() << endl;
        _result = false;
    }
    return _result;
}

bool camera_spk::confset_imgBufSz(int64_t _bufszFrm) {
    bool _result = true;
    try
    {
        //INodeMap & sNodeMap = pCam->getGetStreamNodeMap();
        //CIntegerPtr StreamNode = sNodeMap.GetNode("StreamDefaultBufferCount");
        // int64_t bufferCount = StreamNode->GetValue(); // current buffer size
        //StreamNode->SetValue(_bufszFrm);
    }
    catch (Spinnaker::Exception &e)
    {
        cout << "Error: " << e.what() << endl;
        _result = false;
    }
    return _result;
}

bool camera_spk::confset_ROI(int _w, int _h, int _offsetx, int _offsety) {
    bool _result = true;
    try
    {
        pCam->OffsetX.SetValue(_offsetx);
        offset_x = pCam->OffsetX.GetValue();
        pCam->OffsetY.SetValue(_offsety);
        offset_y = pCam->OffsetY.GetValue();
        pCam->Width.SetValue(_w);
        width = pCam->Width.GetValue();
        pCam->Height.SetValue(_h);
        height = pCam->Height.GetValue();
        ROIsz_byte = width*height*bytePerPx;
        //INodeMap & sNodeMap = pCam->getGetStreamNodeMap();
        //CIntegerPtr StreamNode = sNodeMap.GetNode("StreamDefaultBufferCount");
        // int64_t bufferCount = StreamNode->GetValue(); // current buffer size
        //StreamNode->SetValue(_bufszFrm);
    }
    catch (Spinnaker::Exception &e)
    {
        cout << "Error: " << e.what() << endl;
        _result = false;
    }
    return _result;
}


bool camera_spk::confset_frmRate(float _frmRate){
    bool _result = true;
    try
    {
        INodeMap & nodeMap = pCam->GetNodeMap();
        // Set AcquisitionMode to Continuous | AcquisitionModeEnums {Continuous, SingleFrame, MultiFrame}
        CEnumerationPtr AcquisitionMode = nodeMap.GetNode("AcquisitionMode");
        AcquisitionMode->SetIntValue(AcquisitionMode->GetEntryByName("Continuous")->GetValue());
        double frmRate_max = pCam->AcquisitionFrameRate.GetMax();

        // Command node
        //CCommandPtr ptrTimestampLatch = pCam->GetNodeMap().GetNode("TimestampLatch");
        // Execute command
        //ptrTimestampLatch->Execute();


        //int64_t b = pCam->Timestamp.GetValue();
        //sleep(0.1);
        //int64_t c = pCam->Timestamp.GetValue();
        //int64_t d = pCam->Timestamp.GetValue();



        // Set AcquisitionFrameRateEnable
        CBooleanPtr(nodeMap.GetNode("AcquisitionFrameRateEnable"))->SetValue(true);
        if (_frmRate > frmRate_max)
            _frmRate = frmRate_max;
        CFloatPtr(pCam->GetNodeMap().GetNode("AcquisitionFrameRate"))->SetValue(_frmRate);
        frmRate = pCam->AcquisitionFrameRate.GetValue();
    }
    catch (Spinnaker::Exception &e)
    {
        cout << "Error: " << e.what() << endl;
        _result = false;
    }
    return _result;

    //Command node
    // CCommandPtr ptrTimestampLatch = pCam->GetNodeMap().GetNode("TimestampLatch");
    //Execute command
    // ptrTimestampLatch->Execute();
}


// functions for READING

void camera_spk::image_malloc(void) {
    allImages = (uint8_t **)malloc(imgcntMax * sizeof(uint8_t *));
    for (uint64_t i=0; i< imgcntMax; i++)
        allImages[i] = (uint8_t *)malloc(ROIsz_byte * sizeof(uint8_t));

    testIm = (uint8_t *)malloc(ROIsz_byte * sizeof(uint8_t));
    // allData = (ImgData *)malloc(imgcntMax * sizeof(ImgData));
}

uint64_t camera_spk::getTimeStamp(void) {
    //return pCam->Timestamp.GetValue();
    return pResultImage->GetTimeStamp(); // pixel
}

size_t camera_spk::getImgHeight(void) {
    return CIntegerPtr(pCam->GetNodeMap().GetNode("Height"))->GetValue(); // pixel
}

size_t camera_spk::getImgWidth(void) {
    return CIntegerPtr(pCam->GetNodeMap().GetNode("Width"))->GetValue(); // pixel
}

size_t camera_spk::getImgBufByte(void) {
    return pResultImage->GetBufferSize(); // pixel
}

uint64_t camera_spk::getImgFrmID(void) {
    return pResultImage->GetFrameID();
}

void * camera_spk::getImgData(void) {
    return pResultImage->GetData();
}


uint64_t camera_spk::getFrameCount(void) {
    return FrameCount;
}

uint64_t camera_spk::image_getIDSavedData(uint64_t _FrmNo) {
    return _FrmNo%imgcntMax;
}

void camera_spk::cpyToSaveData(ImagePtr _img) {
    uint64_t id = image_getIDSavedData(FrameCount);
    memcpy(allImages[id], _img->GetData(), ROIsz_byte);
    //allData[id] = curData;
}
void camera_spk::cpyToSaveData(void) {
    cpyToSaveData(pResultImage);
}

uint8_t * camera_spk::getSavedImg(uint64_t _FrmNo) {
    return allImages[image_getIDSavedData(_FrmNo)];
}


INodeMap & camera_spk::GetTLDevice(void) {
    // Retrieve TL device nodemap and print device information
    INodeMap & nodeMapTLDevice = pCam->GetTLDeviceNodeMap();
    PrintDeviceInfo(nodeMapTLDevice);
    return nodeMapTLDevice;
}

int camera_spk::PrintDeviceInfo(INodeMap & nodeMap)
{
    int result = 0;

    cout << endl << "*** DEVICE INFORMATION ***" << endl << endl;

    try
    {
        FeatureList_t features;
        CCategoryPtr category = nodeMap.GetNode("DeviceInformation");
        if (IsAvailable(category) && IsReadable(category))
        {
            category->GetFeatures(features);

            FeatureList_t::const_iterator it;
            for (it = features.begin(); it != features.end(); ++it)
            {
                CNodePtr pfeatureNode = *it;
                cout << pfeatureNode->GetName() << " : ";
                CValuePtr pValue = (CValuePtr)pfeatureNode;
                cout << (IsReadable(pValue) ? pValue->ToString() : "Node not readable");
                cout << endl;
            }
        }
        else
        {
            cout << "Device control information not available." << endl;
        }
    }
    catch (Spinnaker::Exception &e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}
