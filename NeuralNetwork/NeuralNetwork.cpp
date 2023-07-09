#ifndef NEURALNETWORK_CPP
#define NEURALNETWORK_CPP

#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork(NetworkSettings Settings)
{
    int lastLayer = Settings.NumberOfLayers - 1;

    Layers = (NetworkLayer **) malloc(Settings.NumberOfLayers*sizeof(NetworkLayer *));
    for(int i = 1; i < lastLayer; i ++)
    {
        Layers[i] = new NetworkLayer(Settings.Layers[i - 1], Settings.Layers[i], Settings.Activation, Settings.Cost);
    }

    Layers[lastLayer] = new NetworkLayer(Settings.Layers[lastLayer - 1], Settings.Layers[lastLayer], Settings.OutputActivation, Settings.Cost);

    settings = Settings;
}

NeuralNetwork::~NeuralNetwork()
{
    for(int i = 1; i < settings.NumberOfLayers; i ++)
    {
        free(Layers[i]);
    }
    free(Layers);
}

double *NeuralNetwork::Eval(double *Inputs)
{
    double *outputs;
    for(int i = 1; i < settings.NumberOfLayers; i++)
    {
        outputs = Layers[i]->EvalOutput(Inputs);
        Inputs = outputs;
    }
    return outputs;
}

void NeuralNetwork::UpdateValues(double *input, double *expected)
{
    double *output;
    output = Eval(input);
    int lastLayer = settings.NumberOfLayers - 1;
    double *derivativeOutputNodeValues = Layers[lastLayer]->DerivativeNodeValues(expected);
    Layers[lastLayer]->UpdateDerivatives(derivativeOutputNodeValues);
    double *derivativeValues = derivativeOutputNodeValues;
    for(int i = lastLayer - 1; i >= 1; i--)
    {
        derivativeValues = Layers[i]->HiddenLayerDerivativeNodeValues(Layers[i + 1]->GetNodes(), derivativeValues, settings.Layers[i + 1]);
        Layers[i]->UpdateDerivatives(derivativeValues);
    }
}


void NeuralNetwork::Learn(Dataset trainingData)
{
    int lastLayer = settings.NumberOfLayers - 1;

    if(settings.BatchSize == 0)
    {
        settings.BatchSize = 1;
    }

    /*
        Make Training Batches
    */
    Dataset **trainingBatches = (Dataset **) malloc(sizeof(Dataset*)*settings.BatchSize);
    for(int i = 0; i < settings.BatchSize; i++)
    {
        trainingBatches[i] = new Dataset();
        trainingBatches[i]->size = trainingData.size/settings.BatchSize;
        trainingBatches[i]->Inputs =            (double **) malloc(sizeof(double *) * trainingBatches[i]->size);
        trainingBatches[i]->ExpectedOutputs =   (double **) malloc(sizeof(double *) * trainingBatches[i]->size);

        for(int j = 0; j < trainingBatches[i]->size; j++)
        {
            trainingBatches[i]->Inputs[j] = (double *) malloc(sizeof(double) * settings.Layers[0]);
            for(int nIn = 0; nIn < settings.Layers[0]; nIn++)
            {
                trainingBatches[i]->Inputs[j][nIn] = trainingData.Inputs[i*trainingBatches[i]->size + j][nIn];
            }


            trainingBatches[i]->ExpectedOutputs[j] = (double *) malloc(sizeof(double) * settings.Layers[lastLayer]);
            for(int nOut = 0; nOut < settings.Layers[lastLayer]; nOut++)
            {
                trainingBatches[i]->ExpectedOutputs[j][nOut] = trainingData.ExpectedOutputs[i*trainingBatches[i]->size + j][nOut];
            }
        }
    }
    ////////

    /*
        THREADING
    */

    std::thread **threadList = (std::thread **) malloc (sizeof(std::thread*) * settings.BatchSize);
    for(int i = 0; i < settings.BatchSize; i++)
    {
        threadList[i] = new std::thread(&NeuralNetwork::LearnThread, this, std::ref(trainingBatches[i]));
    }

    for(int i = 0; i < settings.BatchSize; i++)
    {
        threadList[i]->join();
        free(threadList[i]);

        for(int j = 0; j < trainingBatches[i]->size; j++)
        {
            free(trainingBatches[i]->Inputs[j]);
            free(trainingBatches[i]->ExpectedOutputs[j]);
        }        

        free(trainingBatches[i]->Inputs);
        free(trainingBatches[i]->ExpectedOutputs);

        free(trainingBatches[i]);
    }

    free(threadList);
    free(trainingBatches);

    ///////////////

    
    UpdateDerivatives(settings.LearnRate);

}

void NeuralNetwork::LearnThread(Dataset *trainingData)
{
    for(int i = 0; i < trainingData->size; i++)
    {
        UpdateValues(trainingData->Inputs[i], trainingData->ExpectedOutputs[i]);
    }
}

void NeuralNetwork::UpdateDerivatives(double learnRate)
{
    for(int i = 1; i < settings.NumberOfLayers; i++)
    {
        Layers[i]->ApplyCost(learnRate);
    }
}

void NeuralNetwork::ClearAllDerivativesG()
{
    for(int i = 0; i < settings.NumberOfLayers; i++)
    {
        Layers[i]->ClearDerivativesG();
    }
}

double NeuralNetwork::TotalError(Dataset trainingData)
{
    double *output;
    double error = 0.0;
    int lastLayer = settings.NumberOfLayers - 1;

    for(int i = 0; i < trainingData.size; i++)
    {
        output = Eval(trainingData.Inputs[i]);
        error += Layers[lastLayer]->Error(output, trainingData.ExpectedOutputs[i]);
    }
    return error/trainingData.size;
}


#endif