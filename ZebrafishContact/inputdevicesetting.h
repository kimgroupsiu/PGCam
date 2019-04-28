#ifndef INPUTDEVICESETTING_H
#define INPUTDEVICESETTING_H


#include <QProcess>

class inputdevicesetting
{
public:
    inputdevicesetting();
    void loadDevice();
    void loadEvent();
    QStringList getDev();
    QStringList getEvent();
    QString setDevID(int index);
    QString setDevID(QString str);
    QString getDevID();
    QString setEventID(int index);
    QString setEventID(QString str);
    QString getEventID();
    void setEnable(bool flag);
    void init();
    bool isset();
    void setPassword(const char * _pw);
private:
    QString password;
    QProcess proc;
    QString stdout;
    QStringList listIDFull;
    QStringList listID;
    QStringList listEventFull;
    QStringList listEvtID;
    QString DevID;
    QString EvtID;
    bool isset_DevID;
    bool isset_EvtID;
};

#endif // INPUTDEVICESETTING_H
