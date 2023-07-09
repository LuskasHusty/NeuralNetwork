#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include "NetworkSettings.hpp"
#include "NetworkLayer.hpp"

#include <stdlib.h>
#include <thread>

class NeuralNetwork
{
    public:
        NeuralNetwork(NetworkSettings Settings);
        ~NeuralNetwork();

        void Learn(Dataset trainingData);
        double *Eval(double *Inputs);
        void UpdateValues(double *input, double *expected);
        void UpdateDerivatives(double learnRate);
        void ClearAllDerivativesG();
        double TotalError(Dataset trainingData);

        NetworkNode **GetNetwork();
        void SetNetwork(NetworkNode **Nodes);
        
    private:
        void LearnThread(Dataset *trainingData);

        NetworkLayer **Layers;
        NetworkSettings settings;

        double *derivativeOutputNodeValues;
        double *derivativeCostWeighted;

};

#endif