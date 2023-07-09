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
        double Error(double *output, double *expected);
        void Learn(double learnRate);
        void ApplyCost(double learnRate);
        double CostGradient(int nodeIndex);
        double *DerivativeNodeValues(double *expected);
        void UpdateDerivatives(double *derivativeValues);
        double *HiddenLayerDerivativeNodeValues(NetworkNode *NextLayerNodes, double *NextLayerDerivativeNodeValues, int NextLayerSize);
        void ClearDerivativesG();

        NetworkNode *GetNodes();
        void SetNodes(NetworkNode *Nodes);
    private:

        //double nodeError(double output, double expected);


        NetworkNode *nodes;
        int numOutputs;
        int numInputs;

        double *Output;
        double *Input; // Pointer of the previous Layers Output
        double *WeightedInputs;
        Functions activation;
        Function2D cost;

        double **errorDerivativesG;
        double *biasDerivativesG;

        double *derivativeValues;

        std::mutex BiasGuard;
        std::mutex WeightGuard;
};

#endif