#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "proctime.h"
#include "inputdevice.h"
#include "PointGreyCamera.h"
#include "camera_spk.h"
#include "bolbimgproc.h"

#include "./DSC_GPU/dsp_gpu.h"

#include <stdio.h>
#include <string.h>
using namespace std;

#include <thread>
#include <chrono>  // CLOCK
typedef std::chrono::high_resolution_clock Clock;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void on_pushButton_cam_connect_clicked();

    void on_pushButton_sensor_connect_clicked();

    void on_pushButton_connect_clicked();

    void on_pushButton_record_clicked();

    void on_pushButton_cam_update_clicked();

    void on_pushButton_improc_update_clicked();

    void on_checkBox_improc_enable_clicked();

private:
    Ui::MainWindow *ui;
    inputdevice sensor;
    PointGreyCamera cam;
    camera_spk cam2;

    bolbimgproc improc;
    DSP_gpu gpuproc;

    // process boolean function
    bool g_bProcessRunning;
    timeval time_org; bool time_init;


    // timer functions - Display
    void onTimer();
    int id_timer;
    virtual void timerEvent( QTimerEvent*) override { onTimer(); };
    QImage * im_disp;
    void ImageGraytoRGB32(uchar * src);
    proctime display_cycle_ms;
    timeval time_disp;

    // Thread
    unique_ptr<std::thread> m_pThreadImage;
    void Image_Thread(void); bool m_bThreadImageEnable;
    proctime cam_cycle_ms, Image_ProcTime_ms;

    unique_ptr<std::thread> m_pThreadSensor;
    void Sensor_thread(void); bool m_bThreadSensorEnable;
    proctime Sensor_cycle_ms;

    // Saving
    bool b_isrecording; timeval time_savebegin; uint64_t recFrmNoBegin;
    std::thread m_pThreadSave;
    void save_thread(void);
};

#endif // MAINWINDOW_H
