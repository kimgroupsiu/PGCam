#ifndef CAMERA_SPK_H
#define CAMERA_SPK_H

#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"

using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace Spinnaker::GenICam;
using namespace std;


class camera_spk
{
public:
    camera_spk();
    ~camera_spk();
    bool isReady;
    bool openCamera(void);
    void closeCamera(void);
    int initCamera(void);
    bool grabImage(void);
    int beginAcquisition(void);
    int stopAcquisition(void);

    // setting
    bool confset(void);
    bool confset_exptime(double _tims_us);
    bool confset_gain(double _gain);
    bool confset_gamma(double _gamma);
    bool confset_frmRate(float _frmRate);
    bool confset_xRev(bool);
    bool confset_yRev(bool);

    bool confset_ROI(int _w, int _h, int _offsetx, int offsety);
    bool confset_imgBufSz(int64_t _bufszFrm);

    // initial setting
    void setSerialNo(char * _no);
    void setPassword(char * _no);

    void image_malloc(void);

    // reading
    CameraPtr pCam;
    size_t getImgWidth(void);
    size_t getImgHeight(void);
    size_t getImgBufByte(void);
    uint64_t getImgFrmID(void);
    void * getImgData(void);
    uint64_t getTimeStamp(void);
    uint64_t image_getIDSavedData(uint64_t _FrmNo);
    void cpyToSaveData(ImagePtr _img);
    void cpyToSaveData(void);


    uint64_t getFrameCount(void);
    uint8_t * getSavedImg(uint64_t _FrmNo);

    double frmRate;
    double frmRate_max;
    double exptime_us;
    double exptime_us_max;
    double gain;
    double gain_max;
    double gamma;
    double gamma_max;
    unsigned int offset_x;
    unsigned int offset_y;
    unsigned int width;
    unsigned int height;

    uint8_t ** allImages;
    uint8_t * testIm;
    unsigned int imgcntMax;
    unsigned int ROIsz_byte;
    unsigned int bytePerPx;
    uint64_t FrameCount;

private:
    void setJetsonTX2_usbbandwidth(void);
    unsigned int loadCamlist_Spinnaker(void);
    INodeMap & GetTLDevice(void);
    int PrintDeviceInfo(INodeMap & nodeMap);

    SystemPtr sys;
    CameraList camList;
    unsigned int numCameras;
    char serial[256];
    char password[256];
    ImagePtr pResultImage;

};

#endif // CAMERA_SPK_H
