#ifndef NETWORKLAYER_CPP
#define NETWORKLAYER_CPP

#include "NetworkLayer.hpp"

NetworkLayer::NetworkLayer(int Inputs, int Outputs, Functions Activation, Function2D Cost)
{
    srand(time(NULL));

    numInputs = Inputs;
    numOutputs = Outputs;

    errorDerivativesG = (double **) malloc(numOutputs*sizeof(double *));
    biasDerivativesG = (double *) malloc(numOutputs*sizeof(double));

    bVels = (double *) malloc(numOutputs*sizeof(double));
    wVels = (double **) malloc(numOutputs*sizeof(double *));

    nodes = (NetworkNode *) malloc(numOutputs*sizeof(NetworkNode));
    for(int i = 0; i < numOutputs; i++)
    {
        errorDerivativesG[i] = (double *) malloc(numInputs*sizeof(double));
        wVels[i] = (double *) malloc(numInputs*sizeof(double));
        biasDerivativesG[i] = 0;
        bVels[i] = 0;

        nodes[i].Bias = 0.0;
        nodes[i].Weight = (double *) malloc(numInputs*sizeof(double));
        for(int j = 0; j < numInputs; j++)
        {
            wVels[i][j] = 0;
            errorDerivativesG[i][j] = 0;
            nodes[i].Weight[j] = (double)((rand() % 3) - 1)/sqrt(numInputs);
        }
    }

    activation = Activation;
    cost = Cost;

    WeightedInputs = (double *) malloc(numOutputs*sizeof(double));

    derivativeValues = (double *) malloc(sizeof(double)*numOutputs);
}

NetworkLayer::~NetworkLayer()
{
    free(nodes);
    free(WeightedInputs);

    for(int i = 0; i < numOutputs; i++)
    {
        free(wVels[i]);
        free(errorDerivativesG[i]);
        free(nodes[i].Weight);
    }

    free(wVels);
    free(bVels);

    free(errorDerivativesG);
    free(biasDerivativesG);

    free(derivativeValues);
}

double *NetworkLayer::EvalOutput(double *Inputs)
{
    double *Output = (double *) malloc(sizeof(double)*numOutputs); 

    for(int i = 0; i < numOutputs; i++)
    {
        WeightedInputs[i] = nodes[i].Bias;
        for(int j = 0; j < numInputs; j++)
        {
            WeightedInputs[i] += Inputs[j]*nodes[i].Weight[j];
        }


        //Calculate Activations
        Output[i] = activation.Function(WeightedInputs[i]);
        //std::cout << Output[i] << '\n';
    }

    return Output;
}

double NetworkLayer::Error(double *output, double *expected)
{
    double error = 0.0;
    for(int i = 0; i < numOutputs; i++)
    {
        error += cost.Function(output[i], expected[i]);
    }
    return error;
}

void NetworkLayer::ApplyCost(double learnRate, double momentum, double regularization)
{
    double wDecay = (1 - (regularization * learnRate));
    for(int i = 0; i < numOutputs; i++)
    {
        double bVel = (bVels[i]*momentum) - (biasDerivativesG[i] * learnRate);
        bVels[i] = bVel;
        nodes[i].Bias += bVel;
        //nodes[i].Bias -=  biasDerivativesG[i] * learnRate;
        
        for(int j = 0; j < numInputs; j++)
        {
            double wVel = (wVels[i][j]*momentum) - (errorDerivativesG[i][j] * learnRate);
            wVels[i][j] = wVel;
            nodes[i].Weight[j] = (nodes[i].Weight[j] * wDecay) + wVel;
            //nodes[i].Weight[j] -= errorDerivativesG[i][j] * learnRate;

            //Clear Derivatives
            errorDerivativesG[i][j] = 0;
        }

        //Clear Derivatives
        biasDerivativesG[i] = 0;
    }
}

double *NetworkLayer::DerivativeNodeValues(double *expected, double *output)
{
    for(int i = 0; i < numOutputs; i++)
    {
        double errorDerivative = cost.Derivative(output[i], expected[i]);
        double activationDerivative = activation.Derivative(WeightedInputs[i]);
        derivativeValues[i] = activationDerivative * errorDerivative;
    }
    return derivativeValues;
}

void NetworkLayer::UpdateDerivatives(double *DerivativeValues, double *input)
{
    std::unique_lock<std::mutex> wLock(WeightGuard);
    for(int i = 0; i < numOutputs; i++)
    {
        for(int j = 0; j < numInputs; j++)
        {
            errorDerivativesG[i][j] += input[j]*DerivativeValues[i];
        }
    }
    wLock.unlock();

    std::unique_lock<std::mutex> bLock(BiasGuard);
    for(int i = 0; i < numOutputs; i++)
    {
        biasDerivativesG[i] += 1 * DerivativeValues[i];
    }
    bLock.unlock();
}   

double *NetworkLayer::HiddenLayerDerivativeNodeValues(NetworkNode *NextLayerNodes, double *NextLayerDerivativeNodeValues, int NextLayerSize)
{
    for(int i = 0; i < numOutputs; i++)
    {
        double derivativeValue = 0.0;
        for(int j = 0; j < NextLayerSize; j++)
        {
            derivativeValue += NextLayerNodes[j].Weight[i] * NextLayerDerivativeNodeValues[j];
        }
        derivativeValue *= activation.Derivative(WeightedInputs[i]);
        derivativeValues[i] = derivativeValue;
    }
    return derivativeValues;
}

NetworkNode *NetworkLayer::GetNodes()
{
    return nodes;
}

void NetworkLayer::SetNodes(NetworkNode *Nodes)
{
    for(int i = 0; i < numOutputs; i++)
    {
        nodes[i].Bias =  Nodes[i].Bias;
        
        for(int j = 0; j < numInputs; j++)
        {
            nodes[i].Weight[j] = Nodes[i].Weight[j];
        }
    }
}

void NetworkLayer::ClearDerivativesG()
{
    for(int i = 0; i < numOutputs; i++)
    {
        for(int j = 0; j < numInputs; j++)
        {
            errorDerivativesG[i][j] = 0.0;
        }
        biasDerivativesG[i] = 0.0;
    }
}


#endif