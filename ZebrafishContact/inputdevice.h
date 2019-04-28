#ifndef INPUTDEVICE_H
#define INPUTDEVICE_H

#include "inputdevicesetting.h"
#include <linux/input.h>
#include <fstream>
#include <iostream>

//enum TPcode {X, Y, CX, CY, lenMajor, lenMinor, CX2, CY2, lenMajor2, lenMinor2, TrkID} ;
enum TPcode {CX, CY, TrkID, CX2, CY2, TrkID2} ;
struct TouchPadData {
    struct timeval time;
    int32_t data[6];
};

class inputdevice
{
public:
    inputdevice();
    ~inputdevice();
    // Settings
    void load();
    bool init(int id_dev, int id_evt);
    bool init(QString str);
    bool connect();
    inputdevicesetting setting;
    int fd;
    uint64_t cntMax;

    // Raw Reading Data & Setting
    struct input_event ev;
    std::vector<struct input_event> ev_all;
    int readtype;
    bool readout();
    void setType(int _t);
    struct input_event * getfirstEv(void);
    struct input_event * getlastEv(void);
    bool removefirstEv(void);
    int getcnt_ev(void);
    void setcntMax(uint64_t _cnt);



    // Data - zForce
    TouchPadData DispData;
    TouchPadData curData;
    std::vector<struct TouchPadData> allData;
    timeval tset;
    int slot_offset;
    void clearCurDataOnly(void);
    void clearCurData(void);
    bool updateToCurData(struct input_event _v);
    void updateDispData(void);
    void deleteDispData(int _offset);
    TouchPadData * getfirstData(void);
    bool removefirstData(void);
    void setdtMin(uint32_t _us);
    int getcnt(void);

    // Frame Data
    std::vector<uint64_t> frmNo;
    void pushfrmNo(uint64_t _frmNo);
    uint64_t getfirstfrmNo(void);
    uint64_t getlastfrmNo(void);
    bool removefirstfrmNo(void);

    //
    bool updateVec(void);

};



//ABS_X			0x00 (0)
//ABS_Y			0x01 (1)
//ABS_MT_SLOT           0x2f (47)   /* MT slot being modified */
//ABS_MT_TOUCH_MAJOR	0x30 (48)	/* Major axis of touching ellipse */
//ABS_MT_TOUCH_MINOR	0x31 (49)	/* Minor axis (omit if circular) */
//ABS_MT_POSITION_X	0x35	(53)/* Center X touch position */
//ABS_MT_POSITION_Y	0x36	(54)/* Center Y touch position */
//ABS_MT_TRACKING_ID	0x39	(57)/* Unique ID of initiated contact */



/* example
 *
 * // load device names and event
 * mouse1.load(); // load all devices and events
 * ui->comboBox_DevID->addItems(mouse1.setting.getDev()); // display in combobox for selection
 * ui->comboBox_DevEvent->addItems(mouse1.setting.getEvent()); // display in combobox for selection
 *
 * // select device
 * bool suc = mouse1.init(ui->lineEdit->text()); // option1. select from the keyword
 * bool suc = mouse1.init(ui->comboBox_DevID->currentIndex(), ui->comboBox_DevEvent->currentIndex()); // option2. select from the combobox.
 *
 * // setting the maximum count & reading type
 * mouse1.setcntMax(500);
 * mouse1.setType(EV_REL);
 *
 * // read
 * while(b_gXXXX) {
 *  mouse1.readout(); // read event
 *  mouse1.pushfrmNo(frmNo); // update frmNo
 *  mouse1.updateVec(); // update the vector based on the maxCnt (maxCnt==0 will set the no-maximum)
 * }
 *
 * // save
 * while (mouse1.ev_all.size() > 0){
 *  outfile.write((char *)mouse1.getfirstData(), mouse1.svdata_sz);
 *  mouse1.removefirstData();
 * }
*/
#endif // INPUTDEVICE_H
