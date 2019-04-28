#include "inputdevicesetting.h"

inputdevicesetting::inputdevicesetting()
{
    password = QString("nvidia");
    isset_DevID = false;
    isset_EvtID = false;
}
void inputdevicesetting::setPassword(const char * _pw) {
    password = QString(_pw);
}

void inputdevicesetting::loadDevice() {
    proc.start("xinput --list");
    proc.waitForFinished(-1); // wait until finished
    stdout = proc.readAllStandardOutput();
    listIDFull= stdout.split("\n");
    proc.start("xinput --list --id-only");
    proc.waitForFinished(-1); // wait until finished
    stdout = proc.readAllStandardOutput();
    listID = stdout.split("\n");
    DevID = QString();
    isset_DevID = false;
}

QStringList inputdevicesetting::getDev() {
    return listIDFull;
}

QString inputdevicesetting::setDevID(int index) {
    isset_DevID = true;
    return DevID = listID[index];
}

QString inputdevicesetting::setDevID(QString str) {
    for (int i = 0; i < listIDFull.size(); i++) {
        int pt = listIDFull[i].indexOf(str);
        if (pt > 0){
            isset_DevID= true;
            DevID = listID[i];
            return DevID;
        }
    }
    return DevID = QString();
}

QString inputdevicesetting::getDevID() {
    if (isset_DevID)
        return DevID;
    else
        return QString();
}

void inputdevicesetting::loadEvent() {
    proc.start("cat /proc/bus/input/devices");
    proc.waitForFinished(-1); // wait until finished
    stdout = proc.readAllStandardOutput();
    listEventFull= stdout.split("\n\n");
    listEvtID.clear();
    for (int i = 0; i < listEventFull.size(); i++) {
        int pt1 = listEventFull[i].indexOf("event")+5;
        int pt2 = listEventFull[i].indexOf("\n", pt1);
        QString st = listEventFull[i].mid(pt1, pt2-pt1-1);
        listEvtID.append(st);
    }
    EvtID = QString();
    isset_EvtID = false;
}

QStringList inputdevicesetting::getEvent() {
    return listEventFull;
}

QString inputdevicesetting::setEventID(int index) {
    isset_EvtID= true;
    return EvtID = listEvtID[index];
}


QString inputdevicesetting::setEventID(QString str) {

    for (int i = 0; i < listEventFull.size(); i++) {
        int pt = listEventFull[i].indexOf(str);
        if (pt > 0){
            isset_EvtID= true;
            EvtID = listEvtID[i];
            return EvtID;
        }
    }
    EvtID = QString();
    return EvtID;
}

QString inputdevicesetting::getEventID() {
    if (isset_EvtID)
        return EvtID;
    else
        return QString();
}



void inputdevicesetting::setEnable(bool flag){
    char cmd[256];
    if (isset_DevID) {
        if (flag) {
            sprintf(cmd, "echo %s | sudo -S xinput set-int-prop %s \"Device Enabled\" 8 1", password.toStdString().c_str(), DevID.toStdString().c_str());
        }
        else {
            sprintf(cmd, "echo %s | sudo -S xinput set-int-prop %s \"Device Enabled\" 8 0", password.toStdString().c_str(), DevID.toStdString().c_str());
        }
        system(cmd);
    }
}

void inputdevicesetting::init(){
    char cmd[256];
    if (isset_EvtID) {
        sprintf(cmd, "echo %s | sudo -S chmod 777 /dev/input/event%s", password.toStdString().c_str(), EvtID.toStdString().c_str());
        system(cmd);
    }
}

bool inputdevicesetting::isset() {
    return isset_DevID && isset_EvtID;
}
