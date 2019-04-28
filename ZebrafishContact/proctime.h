#ifndef PROCTIME_H
#define PROCTIME_H

#include <vector>

class proctime
{
public:
    proctime();
    ~proctime();

    std::vector<double> v;
    int countmax;
    double vmax;
    double vmean;
    double update(double _new);
    double getMax(void);
    double getMean(void);
    void reset(void);
private:
    double updateMax(void);
    double updateMean(void);
};

#endif // PROCTIME_H
