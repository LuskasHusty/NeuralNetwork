#ifndef NETWORKSETTINGS_CPP
#define NETWORKSETTINGS_CPP

#include "NetworkSettings.hpp"

double Default_ActivationSigmoid(double x)
{
    return 1/ (1 + exp(-x));
}

double Default_ActivationSigmoidDerivative(double x)
{
    double out = Default_ActivationSigmoid(x);
    return out * (1 - out);
}

double Default_CostMeanSquareError(double x, double y)
{
    double error = x - y;
    return error*error;
}
double Default_CostMeanSquareErrorDerivative(double x, double y)
{
    return 2*(x - y);
}

#endif