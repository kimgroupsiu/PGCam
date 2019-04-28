#include "inputdevice.h"
#include <X11/Xlib.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/time.h>

inputdevice::inputdevice()
{
    fd = -2;
    readtype = -1;
    cntMax = 0;
    allData.clear();
    setdtMin(5000);
    clearCurData();
}

inputdevice::~inputdevice() {
}


void inputdevice::load() {
    setting.loadDevice();
    setting.loadEvent();
}

bool inputdevice::init(int id_dev, int id_evt) {
    setting.setDevID(id_dev);
    setting.setEventID(id_evt);
    return connect();
}

bool inputdevice::init(QString str) {
    setting.setDevID(str);
    setting.setEventID(str);
    return connect();
}

bool inputdevice::connect() {
    if (!setting.isset())
        return false;
    setting.init();
    setting.setEnable(false);
    char cmd[256];
    sprintf(cmd, "/dev/input/event%s", setting.getEventID().toStdString().c_str());
    fd = open(cmd, O_RDONLY); // from qstring
    if (fd == -1)
        return false;
    else
        return true;
}

bool inputdevice::readout() {
    bool result = false; // new update output over time?
    read(fd, &ev, sizeof(struct input_event));
    if(ev.type == readtype) {
        ev_all.push_back(ev);
        result = updateToCurData(ev);
    }
    return result;
}

void inputdevice::setType(int _t) {
    readtype = _t;
}

struct input_event * inputdevice::getfirstEv(void) {
    struct input_event * _out = NULL;
    if (ev_all.size()>0)
        _out = (struct input_event * )&*(ev_all.begin());
    return _out;
}



bool inputdevice::removefirstEv() {
    bool flag = false;
    if (ev_all.size()>0){
        ev_all.erase(ev_all.begin());
        flag = true;
    }
    return flag;
}

struct input_event * inputdevice::getlastEv(void) {
    struct input_event * _out = NULL;
    if (ev_all.size()>0)
        _out = (struct input_event * )&(ev_all.back());
    return _out;
}

void inputdevice::pushfrmNo(uint64_t _frmNo){
    frmNo.push_back(_frmNo);
}


uint64_t inputdevice::getfirstfrmNo(void){
    if (frmNo.size()>0)
        return frmNo[0];
    else
        return -1;
}


uint64_t inputdevice::getlastfrmNo(void){
    if (frmNo.size()>0)
        return *frmNo.end();
    else
        return -1;
}


bool inputdevice::removefirstfrmNo(void){
    bool flag = false;
    if (frmNo.size()>0){
        frmNo.erase(frmNo.begin());
        flag = true;
    }
    return flag;
}

bool inputdevice::updateVec(void) {
    bool flag = false;
    if (cntMax > 0) {
        if (ev_all.size() > cntMax) {
            flag = flag || removefirstEv();
        }        
        if (allData.size() > cntMax) {
            flag = flag || removefirstData();
        }
        if (frmNo.size() > cntMax) {
            flag = flag || removefirstfrmNo();
        }
    }
    return flag;
}

void inputdevice::setcntMax(uint64_t _cnt) {
    cntMax = _cnt;
}

void inputdevice::setdtMin(uint32_t _us) {
    tset.tv_sec = 0;
    tset.tv_usec = _us;
}

int inputdevice::getcnt(void) {
    return allData.size();
}

int inputdevice::getcnt_ev(void) {
    return ev_all.size();
}

void inputdevice::clearCurData() {
    timerclear(&curData.time);
    for (int i = 0; i<6; i++)
        curData.data[i] = -1;
    slot_offset = 0;
}

void inputdevice::clearCurDataOnly() {
    curData.data[0] = -1;
    curData.data[1] = -1;
    curData.data[3] = -1;
    curData.data[4] = -1;
}



bool inputdevice::updateToCurData(struct input_event _v) {
    bool result = false;
    if (_v.type != readtype)
        return result;

    timeval dt;
    timersub(&_v.time, &curData.time, &dt);
    if (timercmp(&dt, &tset, >)) {
        allData.push_back(curData);
        updateDispData();
        clearCurDataOnly();
        curData.time = _v.time;
        result = true;
    }
    if (_v.code == ABS_MT_SLOT) // when the code is ABS_MT_SLOT, the value is 0 or 1 (two object tracking)
    {
        slot_offset = _v.value*3;
    }
    if (_v.value < 0) {// when the movement of TrkID is over, it reports -1 (remove the Display Data)
        deleteDispData(slot_offset);
        return result;
    }

     switch(_v.code) {
        //case ABS_X: curData.data[TPcode::X] = _v.value; break;
        //case ABS_Y: curData.data[TPcode::Y] = _v.value; break;
        //case ABS_MT_TOUCH_MAJOR: curData.data[TPcode::lenMajor+slot_offset] = _v.value; break;
        //case ABS_MT_TOUCH_MINOR: curData.data[TPcode::lenMinor+slot_offset] = _v.value; break;
        case ABS_MT_POSITION_X: curData.data[TPcode::CX+slot_offset] = _v.value; break;
        case ABS_MT_POSITION_Y: curData.data[TPcode::CY+slot_offset] = _v.value; break;
        case ABS_MT_TRACKING_ID: curData.data[TPcode::TrkID+slot_offset] = _v.value; break;
     };
     return result;
}

/*
//ABS_X			0x00 (0)
//ABS_Y			0x01 (1)
//ABS_MT_SLOT           0x2f (47)   // MT slot being modified
//ABS_MT_TOUCH_MAJOR	0x30 (48)	// Major axis of touching ellipse
//ABS_MT_TOUCH_MINOR	0x31 (49)	// Minor axis (omit if circular)
//ABS_MT_POSITION_X     0x35 (53)   // Center X touch position
//ABS_MT_POSITION_Y     0x36 (54)   // Center Y touch position
//ABS_MT_TRACKING_ID	0x39 (57)   // Unique ID of initiated contact
// https://www.kernel.org/doc/Documentation/input/multi-touch-protocol.txt
struct input_event * _item = mouse1.getlastEv();
    switch(_item->code) {
        case ABS_X: Neodata.data[TPcode::X] = _item->value; break;
        case ABS_Y: Neodata.data[TPcode::Y] = _item->value; break;
        case ABS_MT_SLOT: Neodata.data[TPcode::slot] = _item->value; break;
        case ABS_MT_TOUCH_MAJOR: Neodata.data[TPcode::lenMajor] = _item->value; break;
        case ABS_MT_TOUCH_MINOR: Neodata.data[TPcode::lenMinor] = _item->value; break;
        case ABS_MT_POSITION_X: Neodata.data[TPcode::CX] = _item->value; break;
        case ABS_MT_POSITION_Y: Neodata.data[TPcode::CY] = _item->value; break;
        case ABS_MT_TRACKING_ID: Neodata.data[TPcode::TrkID] = _item->value; break;
    };

}*/

void inputdevice::updateDispData() {
    for (int i = 0; i < 6; i++) {
        if (curData.data[i] >= 0) {
            DispData.data[i] = curData.data[i];
        }
    }
}
void inputdevice::deleteDispData(int _offset) {
    for (int i = 0; i < 3; i++) {
        DispData.data[i+_offset] = -1;
    }
}

TouchPadData * inputdevice::getfirstData(void) {
    TouchPadData * _out = NULL;
    if (allData.size()>0)
        _out = (TouchPadData *)&*(allData.begin());
    return _out;
}

bool inputdevice::removefirstData(void) {
    bool flag = false;
    if (allData.size()>0){
        allData.erase(allData.begin());
        flag = true;
    }
    return flag;
}
