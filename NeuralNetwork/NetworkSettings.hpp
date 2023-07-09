#ifndef NETWORKSETTINGS_HPP
#define NETWORKSETTINGS_HPP

#include "math.h"

double Default_ActivationSigmoid(double x);
double Default_ActivationSigmoidDerivative(double x);

double Default_CostMeanSquareError(double x, double y);
double Default_CostMeanSquareErrorDerivative(double x, double y);

struct Functions
{
    double (* Function) (double x);
    double (* Derivative) (double x);
};

struct Function2D
{
    double (* Function) (double x, double y);
    double (* Derivative) (double x, double y);
};

struct Double2D
{
    double x;
    double y;
};

struct Dataset
{
    int size;
    double **Inputs;
    double **ExpectedOutputs;
};

struct NetworkSettings
{
    int NumberOfLayers;
    int *Layers;

    Functions Activation;
    Functions OutputActivation;
    Function2D Cost;

    double LearnRate;

    double Momentum;
    double BatchSize;
    double Regularization;
};


#endif