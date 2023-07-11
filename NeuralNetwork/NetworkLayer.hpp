#ifndef NETWORKLAYER_HPP
#define NETWORKLAYER_HPP

#include "NetworkSettings.hpp"

#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cmath>
#include <mutex>

struct NetworkNode
{
    double Bias;
    double *Weight;
};

class NetworkLayer
{
    public:
        NetworkLayer(int Inputs, int Outputs, Functions Activation, Function2D Cost);
        ~NetworkLayer();
        double *EvalOutput(double *Inputs);
        double *EvalOutput(double *Inputs, double *WeightedInputs);
        double Error(double *output, double *expected);
        void Learn(double learnRate);
        void ApplyCost(double learnRate, double momentum, double regularization);
        double CostGradient(int nodeIndex);
        double *DerivativeNodeValues(double *expected, double *output, double *WeightedInputs);
        double *HiddenLayerDerivativeNodeValues(NetworkNode *NextLayerNodes, double *NextLayerDerivativeNodeValues, int NextLayerSize, double *WeightedInputs);
        void UpdateDerivatives(double *derivativeValues, double *input);
        void ClearDerivativesG();

        NetworkNode *GetNodes();
        void SetNodes(NetworkNode *Nodes);
    private:

        //double nodeError(double output, double expected);


        NetworkNode *nodes;
        int numOutputs;
        int numInputs;

        Functions activation;
        Function2D cost;

        double **errorDerivativesG;
        double *biasDerivativesG;

        double *derivativeValues;

        std::mutex BiasGuard;
        std::mutex WeightGuard;

        double *bVels;
        double **wVels;
};

#endif