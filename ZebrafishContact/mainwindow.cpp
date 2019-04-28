#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QPainter>


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    size_t a = sizeof(timeval);
    ui->setupUi(this);
    g_bProcessRunning = false;
    m_bThreadImageEnable = false;
    m_bThreadSensorEnable = false;    
    time_init = false;

    // Timer (display)
    const int Timer_UPDATE_INTERVAL_MS = 50;
    id_timer = startTimer(Timer_UPDATE_INTERVAL_MS);
    srand(time(NULL));

    // Sensor
    sensor.load(); // load Devices and Events
    ui->comboBox_sensor_DevID->addItems(sensor.setting.getDev()); // update to combobox
    ui->comboBox_sensor_DevEvent->addItems(sensor.setting.getEvent()); // update to combobox

    // Camera setting
    improc.setEnable(ui->checkBox_improc_enable->isChecked());

    // Display initial setting
    ui->label_display->setScaledContents(true);
    im_disp = new QImage(2048,1536,QImage::Format_RGB888);
    im_disp->fill(0);
    ui->label_display->setPixmap(QPixmap::fromImage(*im_disp));

    // recording
    b_isrecording = false;
    recFrmNoBegin = 0;

//    // GPU test
//    float * gpu_A = NULL;
//    float * gpu_B = NULL;
//    float * gpu_C = NULL;
//    size_t sz = 10;
//    float A[10] = {1,2,3,4,5,6,7,8,9,0};
//    float B[10] = {11,12,13,14,15,16,17,18,19,10};
//    float C[10] = {1};

//    cudaError_t err ;

//    err = cudaDeviceReset();

//    err = cudaMalloc((void **)(&gpu_A), sz*sizeof(float));
//    err = cudaMalloc((void **)(&gpu_B), sz*sizeof(float));
//    err = cudaMalloc((void **)(&gpu_C), sz*sizeof(float));

//    err = cudaMemset(gpu_A, 0, sz*sizeof(float));
//    err = cudaMemset(gpu_B, 0, sz*sizeof(float));
//    err = cudaMemset(gpu_C, 0, sz*sizeof(float));

//    err = cudaMemcpy((float*)gpu_A, (const float*)A, sz*sizeof(float), cudaMemcpyHostToDevice);
//    err = cudaMemcpy((float*)gpu_B, (const float*)B, sz*sizeof(float), cudaMemcpyHostToDevice);

//    VecAdd_kernelGPU(gpu_A, gpu_B, gpu_C, (int)sz);

//    err = cudaMemcpy(C, gpu_C, sz*sizeof(float), cudaMemcpyDeviceToHost);

//    cudaFree(gpu_A);
//    cudaFree(gpu_B);
//    cudaFree(gpu_C);
}

MainWindow::~MainWindow()
{
    b_isrecording = false;
    g_bProcessRunning = false;
    if (m_bThreadImageEnable) m_pThreadImage->join();
    if (m_bThreadSensorEnable) m_pThreadSensor->join();
    delete ui;
    delete im_disp;
}

void MainWindow::onTimer() {
    if (g_bProcessRunning) {
        if ((m_bThreadImageEnable>0) && (cam2.getFrameCount() > 2)) {
            // update display
            if (im_disp->width() != cam2.getImgWidth()) {
                delete im_disp;
                im_disp = new QImage((int)cam2.getImgWidth(), (int)cam2.getImgHeight(), QImage::Format_RGB32);
            }
            //ImageGraytoRGB32((uchar*)cam2.getSavedImg(cam2.getFrameCount() - 1));
            //ImageGraytoRGB32((uchar*)cam2.getImgData());
            if (ui->checkBox_improc_disp->isChecked()) {
                if (cam2.testIm)
                    ImageGraytoRGB32((uchar*)cam2.testIm);
            }
            else
                ImageGraytoRGB32((uchar*)cam2.getSavedImg(cam2.getFrameCount() - 1));


            QPainter painter(im_disp);
            painter.setPen(Qt::red);

            float4 pos = gpuproc.getPos();

            int r = (int)sqrt(sqrt(pos.w));
            painter.drawEllipse(QPointF(pos.x, pos.y) ,r,r);
            painter.drawLine(pos.x, pos.y,
                             pos.x + 5*r*cos(pos.z), pos.y + 5*r*sin(pos.z));
            painter.end();
            // other update
            ui->lcdNumber_cam_cycleAvg_ms->display(cam_cycle_ms.getMean());
            ui->lcdNumber_cam_cycleMax_ms->display(cam_cycle_ms.getMax());
            ui->lcdNumber_cam_procAvg_ms->display(Image_ProcTime_ms.getMean());
            ui->lcdNumber_cam_procMax_ms->display(Image_ProcTime_ms.getMax());
            ui->lcdNumber_improc_x->display((double)pos.x);
            ui->lcdNumber_improc_y->display((double)pos.y);
            ui->lcdNumber_improc_theta->display((double)pos.z*180/3.141592);
            ui->lcdNumber_improc_objsize->display((int)pos.w);
        }
        else {
            // if there is no camera image, clear all with white
            im_disp->fill(Qt::white);
        }
        if (m_bThreadSensorEnable) {
            ui->lcdNumber_sensor_bufFilled->display(sensor.getcnt());
            ui->lcdNumber_sensor_x1->display(sensor.DispData.data[0]);
            ui->lcdNumber_sensor_y1->display(sensor.DispData.data[1]);
            ui->lcdNumber_sensor_ID1->display(sensor.DispData.data[2]);
            ui->lcdNumber_sensor_x2->display(sensor.DispData.data[3]);
            ui->lcdNumber_sensor_y2->display(sensor.DispData.data[4]);
            ui->lcdNumber_sensor_ID2->display(sensor.DispData.data[5]);
            ui->lcdNumber_sensor_cycleAvg_ms->display(Sensor_cycle_ms.getMean());
            ui->lcdNumber_sensor_cycleMax_ms->display(Sensor_cycle_ms.getMax());
            // update display
            QPainter painter(im_disp);
            painter.setPen(Qt::blue);
            int r = (int)(im_disp->width()/100);
            painter.drawEllipse(QPointF(sensor.DispData.data[0]*im_disp->width()/65536
                                ,sensor.DispData.data[1]*im_disp->height()/65536),r,r);
            painter.setPen(Qt::darkMagenta);
            painter.drawEllipse(QPointF(sensor.DispData.data[3]*im_disp->width()/65536
                                ,sensor.DispData.data[4]*im_disp->height()/65536),r,r);
            painter.end();
        }
        ui->label_display->setPixmap(QPixmap::fromImage(*im_disp));
        if (time_init) {
            timeval _cur, _dt;
            if (m_bThreadImageEnable)
                _cur = cam.curData.time;
            else if (m_bThreadSensorEnable)
                _cur = sensor.curData.time;
            else
                gettimeofday(&_cur, NULL);
            timersub(&_cur, &time_org, &_dt);
            char _txt[256];
            uint32_t _h = (uint32_t)(_dt.tv_sec/3600);
            uint32_t _m = (uint32_t)((_dt.tv_sec - _h*3600)/60);
            uint32_t _s = _dt.tv_sec - _h*3600 - _m*60;
            sprintf(_txt, "%02d:%02d:%02d", _h, _m, _s);
            ui->label_time_afterstart->setText(QString(_txt));
            if (b_isrecording) {
                ui->lcdNumber_RecFrame->display((int)(cam.getFrameCount() - recFrmNoBegin));
                timersub(&_cur, &time_savebegin, &_dt);
                _h = (uint32_t)(_dt.tv_sec/3600);
                _m = (uint32_t)((_dt.tv_sec - _h*3600)/60);
                _s = _dt.tv_sec - _h*3600 - _m*60;
                sprintf(_txt, "%02d:%02d:%02d", _h, _m, _s);
                ui->label_time_aftersave->setText(QString(_txt));
            }
        }
    }
    timeval _cur, _dt; gettimeofday(&_cur, NULL); timersub(&_cur, &time_disp, &_dt);
    display_cycle_ms.update((double)(_dt.tv_sec*1000) + (double)_dt.tv_usec/1000); //LOOP TIME
    time_disp = _cur;
    ui->lcdNumber_disp_cycleAvg_ms->display(display_cycle_ms.getMean());
    ui->lcdNumber_disp_cycleMax_ms->display(display_cycle_ms.getMax());
}

void MainWindow::ImageGraytoRGB32(uchar * src) {
    unsigned int cols = im_disp->width();
    unsigned int rows = im_disp->height();
    unsigned int step = cols;
    for( int y=0; y<rows; ++y ) {
        uchar * pSource= src;
        QRgb* pDest= reinterpret_cast<QRgb*>( im_disp->scanLine(y) );
        for( int x=0; x<cols; ++x ) {
            if (pSource == NULL) return;
            *pDest++= qRgb( *pSource, *pSource, *pSource );
            ++pSource;
        } // end for x
        src+= step;
    } // end for y
}
void MainWindow::on_pushButton_connect_clicked()
{
    g_bProcessRunning = true;
    if (!time_init) {
        gettimeofday(&time_org, NULL);
        time_init = true;
    }
    on_pushButton_cam_connect_clicked();
    on_pushButton_sensor_connect_clicked();
}

// ========================== Camera ==========================

void MainWindow::on_pushButton_cam_connect_clicked()
{
    g_bProcessRunning = true;
    if (!m_bThreadImageEnable) {

        ui->spinBox_cam_fps->setEnabled(false);
        ui->spinBox_cam_ROI_offX->setEnabled(false);
        ui->spinBox_cam_ROI_offY->setEnabled(false);
        ui->spinBox_cam_ROI_imgX->setEnabled(false);
        ui->spinBox_cam_ROI_imgY->setEnabled(false);

        improc.setEnable(ui->checkBox_improc_enable->isChecked());

        on_pushButton_improc_update_clicked();

        if (!time_init) {
            gettimeofday(&time_org, NULL);
            time_init = true;
        }

        m_pThreadImage.reset(new std::thread([this] {Image_Thread(); }));
        m_bThreadImageEnable = true;
    }
}


void MainWindow::on_pushButton_cam_update_clicked()
{
    cam.setGain((float)ui->doubleSpinBox_cam_gain->value());
    cam.setShutterSpeed((float)(ui->spinBox_cam_exp_us->value())/1000);
}


void MainWindow::on_pushButton_improc_update_clicked()
{
    improc.setMinObjSz(ui->spinBox_improc_minObjSize->value());
    improc.setTheshold(ui->doubleSpinBox_improc_threshold->value());
    gpuproc.setThreshold(ui->doubleSpinBox_improc_threshold->value()*255.0f);
}

void MainWindow::on_checkBox_improc_enable_clicked()
{
    improc.setEnable(ui->checkBox_improc_enable->isChecked());
}

void MainWindow::Image_Thread()
{
    timeval fstarttime, fEndSeconds, fstarttime2, _dt;
    gettimeofday(&fstarttime2, NULL);
    bool suc = false;


    double _frameRate = (float)ui->spinBox_cam_fps->value();
    double _exp_us = (float)(ui->spinBox_cam_exp_us->value());
    double _gain = (float)ui->doubleSpinBox_cam_gain->value();
    int img_x = ui->spinBox_cam_ROI_imgX->value();
    int img_y = ui->spinBox_cam_ROI_imgY->value();
    int offset_x = ui->spinBox_cam_ROI_offX->value();
    int offset_y = ui->spinBox_cam_ROI_offY->value();

    cam.frameRate = _frameRate; cam.setExpTime(_exp_us); cam.setROI(offset_x, offset_y, img_x, img_y); cam.gain = _gain;


    if (!cam2.openCamera())
        return; // fail
    if (cam2.initCamera() < 0)
        return; // fail
    if (!cam2.confset_ROI(img_x, img_y, offset_x, offset_y))
        return;
    if (!cam2.confset_exptime(_exp_us))
        return;
    if (!cam2.confset_gain(_gain))
        return;
    if (!cam2.confset_gamma(1.0))
        return;
    if (!cam2.confset_imgBufSz(500))
        return;
    if (!cam2.confset_frmRate(_frameRate))
        return;
    if (!cam2.confset_xRev(false))
        return;

    cam2.image_malloc();
    improc.setImSize(cam2.getImgWidth(), cam2.getImgHeight(), cam2.bytePerPx);
    gpuproc.init_memset(cam2.getImgWidth(), cam2.getImgHeight());

    if (cam2.beginAcquisition() < 0)
        return; // fail
    else
        suc = true;


    if (suc) {
        ui->label_cam_msg->setText("Cam Connected");
        ui->label_cam_msg_copy->setText("Cam Connected");
    }
    else {
        ui->label_cam_msg->setText("Cam Failed");
        ui->label_cam_msg_copy->setText("Cam Failed");
        return;
    }


    while (g_bProcessRunning) {
        if (cam2.grabImage()){
            // Image Captured
            gettimeofday(&fstarttime, NULL);

            //cam.curData.time = fstarttime;
            // --------------------------------------------------------------
            //improc.copyIm((char*)cam.getImgData());
            //improc.proc();
            gpuproc.proc((unsigned char*)cam2.getImgData());



            gpuproc.testfn(cam2.testIm); // download

            // --------------------------------------------------------------
            //cam.curData.x = improc.x();
            //cam.curData.y = improc.y();
            //cam.curData.th = improc.th();
            //cam.curData.size = (int32_t)improc.ObjSz();
            cam2.cpyToSaveData();
            // compute the process time and cycle
            gettimeofday(&fEndSeconds , NULL);
            timersub(&fstarttime, &fstarttime2, &_dt);
            cam_cycle_ms.update((double)_dt.tv_sec*1000.0 + ((double)_dt.tv_usec/1000.0)); //LOOP TIME
            timersub(&fEndSeconds, &fstarttime, &_dt);
            Image_ProcTime_ms.update((double)(_dt.tv_sec*1000) + (double)_dt.tv_usec/1000); //LOOP TIME
            fstarttime2 = fstarttime;
        }
    }

    if (cam.disconnectCamera()) {
        ui->label_cam_msg->setText("Cam Disconnected");
        ui->label_cam_msg_copy->setText("Cam Disconnected");
    }

    m_bThreadImageEnable = false;
}

// ========================== Sensors ==========================
void MainWindow::on_pushButton_sensor_connect_clicked()
{
    g_bProcessRunning = true;
    if (!m_bThreadSensorEnable) {
        ui->lineEdit_sensor_namekeyword->setEnabled(false);
        ui->lineEdit_sensor_SN->setEnabled(false);
        ui->spinBox_sensor_bufsz->setEnabled(false);
        ui->spinBox_sensor_cyclemin_us->setEnabled(false);

        if (!time_init) {
            gettimeofday(&time_org, NULL);
            time_init = true;
        }

        m_pThreadSensor.reset(new std::thread([this] {Sensor_thread(); }));
        m_bThreadSensorEnable = true;
    }
}

void MainWindow::Sensor_thread()
{
    timeval fstarttime, fstarttime2, _dt;
    gettimeofday(&fstarttime2, NULL);

    bool suc = false;
    if (ui->checkBox_sensor_selectbyname->isChecked())
        suc = sensor.init(ui->lineEdit_sensor_namekeyword->text());
    else
        suc = sensor.init(ui->comboBox_sensor_DevID->currentIndex(), ui->comboBox_sensor_DevEvent->currentIndex());

    if (suc) {
        ui->label_sensor_msg->setText("Sensor Connected");
        ui->label_sensor_msg_copy->setText("Sensor Connected");
    }
    else {
        ui->label_sensor_msg->setText("Sensor Failure");
        ui->label_sensor_msg_copy->setText("Sensor Failure");
        return;
    }

    sensor.setcntMax(ui->spinBox_sensor_bufsz->value());
    sensor.setdtMin((uint32_t)ui->spinBox_sensor_cyclemin_us->value());
    sensor.setType(EV_ABS);
    sensor.clearCurData();

    while (g_bProcessRunning) {
        if (sensor.readout()) {
            // update time
            gettimeofday(&fstarttime, NULL);
            timersub(&fstarttime, &fstarttime2, &_dt);
            Sensor_cycle_ms.update((double)_dt.tv_sec*1000.0 + ((double)_dt.tv_usec/1000.0)); //LOOP TIME
            fstarttime2 = fstarttime;
        }
        sensor.updateVec();

    }

    m_bThreadSensorEnable = false;
}

// ========================== Saving ==========================
void MainWindow::on_pushButton_record_clicked()
{

    if (g_bProcessRunning) {
        if (b_isrecording) {
            b_isrecording = false;
            ui->progressBar_save->setValue(0);ui->progressBar_save->setTextVisible(false);
            ui->pushButton_record->setText("Record");
        }
        else {
            b_isrecording = true;
            m_pThreadSave = std::thread([this]{save_thread();}); //
            m_pThreadSave.detach();
            ui->progressBar_save->setValue(100); ui->progressBar_save->setTextVisible(true);
            ui->pushButton_record->setText("Stop");
        }
    }
}

void MainWindow::save_thread() {
    ofstream OutFile_img, OutFile_imgData, OutFile_sensor, OutFile_sensorRaw;
    char NAME[256];
    time_t t; struct tm * now; t = time(0); now = localtime(&t);// get time now
    bool isRec_img = ui->checkBox_cam_recImageEnable->isChecked();
    bool isRec_imgData = ui->checkBox_cam_recImageProcDataEnable->isChecked();
    bool isRec_sensor = ui->checkBox_sensor_recPosEnable->isChecked();
    bool isRec_sensorRaw = ui->checkBox_sensor_recRawEnable->isChecked();
    sensor.setcntMax(0); // make the sensor continue to save data
    gettimeofday(&time_savebegin, NULL);
    uint64_t recFrmNo = cam.getFrameCount();
    {
        bool _suc = false;
        ImgData * temp = cam.getSavedImgData(recFrmNo);
        for (int i = 0; i < cam.imgcntMax; i++) {
            if (timercmp(&(temp->time), &time_savebegin, >))
                temp = cam.getSavedImgData(--recFrmNo);
            else {
                _suc = true; break;
            }
        }
        if (!_suc) {
            ui->label_cam_msg->setText("FAIL REC");
            ui->label_cam_msg_copy->setText("FAIL REC");
            ui->label_sensor_msg->setText("FAIL REC");
            ui->label_sensor_msg_copy->setText("FAIL REC");
            return;
        }
        else
            recFrmNoBegin = recFrmNo;
    }
    if (isRec_img) {
        sprintf(NAME, "ZFTrk_%04d%02d%02d_%02d%02d%02d_img.bin", now->tm_year + 1900, (now->tm_mon + 1), now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec);
        OutFile_img.open(NAME, ios::out | ios::binary);
    }
    if (isRec_imgData) {
        sprintf(NAME, "ZFTrk_%04d%02d%02d_%02d%02d%02d_imgdata.bin", now->tm_year + 1900, (now->tm_mon + 1), now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec);
        OutFile_imgData.open(NAME, ios::out | ios::binary);
        struct ImgData temp; temp.time = time_savebegin;
        temp.FrameNo = (int32_t)recFrmNo; temp.size = (int32_t)cam.frameRate; temp.x = (float)cam.getcols(); temp.y = (float)cam.getrows();
        OutFile_imgData.write((char *)&temp, sizeof(struct ImgData));
    }
    if (isRec_sensor) {
        sprintf(NAME, "ZFTrk_%04d%02d%02d_%02d%02d%02d_sensor.bin", now->tm_year + 1900, (now->tm_mon + 1), now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec);
        OutFile_sensor.open(NAME, ios::out | ios::binary);
        struct TouchPadData temp; temp.time = time_savebegin;
        OutFile_sensor.write((char *)&temp, sizeof(struct TouchPadData));
    }
    if (isRec_sensorRaw) {
        sprintf(NAME, "ZFTrk_%04d%02d%02d_%02d%02d%02d_sensorRaw.bin", now->tm_year + 1900, (now->tm_mon + 1), now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec);
        OutFile_sensorRaw.open(NAME, ios::out | ios::binary);
        struct input_event temp; temp.time= time_savebegin;
        OutFile_sensorRaw.write((char *)&temp, sizeof(struct input_event));
    }

    while (b_isrecording) {
    // Image & Image-Data Saving
        if (recFrmNo < cam.getFrameCount()-1) {
            if (isRec_imgData) {
                ImgData * temp = cam.getSavedImgData(recFrmNo);
                OutFile_imgData.write((char *)temp, sizeof(struct ImgData));
            }
            if (isRec_img) {
                uint8_t * temp = cam.getSavedImg(recFrmNo);
                OutFile_img.write((char *)temp, cam.ROIsz_byte);
            }
            recFrmNo++;
        }
    // sensor Data Saving
        if (isRec_sensor && (sensor.getcnt() > 1)) {
            struct TouchPadData * temp = sensor.getfirstData();
            if (timercmp(&(temp->time), &time_savebegin, >))
                OutFile_sensor.write((char *)temp, sizeof(struct TouchPadData));
            sensor.removefirstData();
        }
    // sensor Data Saving
        if (isRec_sensorRaw && (sensor.getcnt_ev() > 1)) {
            struct input_event * temp = sensor.getfirstEv();
            if (timercmp(&(temp->time), &time_savebegin, >))
                OutFile_sensorRaw.write((char *)temp, sizeof(struct input_event));
            sensor.removefirstEv();
        }
    }
    sensor.setcntMax(ui->spinBox_sensor_bufsz->value()); // set the sensor buffer again
    if (isRec_img) OutFile_img.close();
    if (isRec_imgData) OutFile_imgData.close();
    if (isRec_sensor) OutFile_sensor.close();
    if (isRec_sensorRaw) OutFile_sensorRaw.close();
}
