using System;
using System.Collections.Generic;
using System.Linq;

namespace MNNL.Train.Solver
{
    public class GD : ISolver
    {
        private readonly Func<double, double, double> errFunc;
        private readonly Func<double, double, double> gradErrFunc;
        private readonly Dictionary<IConnection, double> con2Grad;
        private readonly Dictionary<IConnection, double> con2GradSumSquared;

        public LearningRegularization Regularization { get; private set; }
        public INetwork Network { get; private set; }
        public LearningCriterion Criterion { get; set; }
        public double Error { get; private set; }
        public double LearningRate { get; set; }
        public bool AdaGrad { get; set; }

        public GD(INetwork network, LearningCriterion criterion = LearningCriterion.MSE,
            LearningRegularization regularization = LearningRegularization.None)
        {
            con2Grad = new Dictionary<IConnection, double>();
            con2GradSumSquared = new Dictionary<IConnection, double>();
            Criterion = criterion;
            Regularization = regularization;
            Network = network;
            LearningRate = .1;
            AdaGrad = true;

            switch (criterion)
            {
                case LearningCriterion.CrossEntropy:
                    errFunc = (desired, estimate) => -desired * Math.Log(estimate) - (1-desired) * Math.Log(1-estimate);
                    gradErrFunc = (desired, estimate) => (1 - estimate) * estimate * (desired - estimate);
                    break;
                default:
                    errFunc = (desired, estimate) =>
                    {
                        var error = desired - estimate;
                        return .5*error*error;
                    };
                    gradErrFunc = (desired, estimate) => -(desired - estimate);
                    break;
            }

            switch (regularization)
            {
                case LearningRegularization.L1Norm:
                    break;
                case LearningRegularization.L2Norm:
                    break;
                case LearningRegularization.Dropout:
                    break;
                default:
                    break;
            }
        }

        public void RunEpoch(double[][] inputs, double[][] desiredOuputs)
        {
            con2Grad.Clear();
            var N = inputs.Length;
            var sumError = new double[Network.NumberOfOutputs];
            var lastLayerNodes = Network.StagedNodes.Last();

            for (var i = 0; i < N; i++)
            {
                //forward propagation
                Network.Forward(inputs[i]);

                for (var j = 0; j < Network.NumberOfOutputs; j++)
                {
                    double noj = Network.Output[j], doij = desiredOuputs[i][j];
                    sumError[j] += errFunc(doij, noj);
                    lastLayerNodes[j].Backward(gradErrFunc(doij, noj));
                }

                //back propagation
                //skip last layer
                for (var j = Network.StagedNodes.Length - 2; j >= 0; j--)
                {
                    var layer = Network.StagedNodes[j];

                    foreach (var node in layer)
                    {
                        foreach (var connection in node.OutConnections)
                        {
                            if (!con2Grad.ContainsKey(connection)) con2Grad.Add(connection, 0.0);

                            con2Grad[connection] += connection.GradientOfParameter()/N;
                        }

                        node.Backward();
                    }
                }
            }

            if(AdaGrad && con2GradSumSquared.Count == 0)
                foreach (var k in con2Grad.Keys)
                {
                    con2GradSumSquared[k] = 0;
                }

            Error = sumError.Sum(s => s/N);
        }

        public void UpdateParameters()
        {
            foreach (var con in con2Grad)
            {
                var lr = LearningRate;

                if (AdaGrad)
                {
                    con2GradSumSquared[con.Key] += (con.Value*con.Value);
                    lr /= Math.Sqrt(con2GradSumSquared[con.Key]);
                }
                
                con.Key[0] += -lr * con.Value;
            }
        }

        public double LossFunc(double desiredOutput, double estimatedOutput)
        {
            return errFunc(desiredOutput, estimatedOutput);
        }
    }
}
