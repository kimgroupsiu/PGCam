#include "proctime.h"
#include <algorithm>

proctime::proctime()
{
    countmax = 100;
    v.clear();
    vmax = 0;
    vmean = 0;
}

proctime::~proctime()
{
    v.clear();
}

double proctime::update(double _new) {
    v.push_back(_new);
    if (v.size()>countmax) {
        // delete first
        v.erase(v.begin());
        updateMax();
        updateMean();
    }
}

double proctime::updateMax(void) {
    return vmax = *(std::max_element(v.begin(), v.end()));
}

double proctime::updateMean(void){
    return vmean = std::accumulate(v.begin(), v.end(), 0.0)/v.size();
}

double proctime::getMax(void) {
    return vmax;
}

double proctime::getMean(void) {
    return vmean;
}

void proctime::reset(void) {
    v.clear();
}
