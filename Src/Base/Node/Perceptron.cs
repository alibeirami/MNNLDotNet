using System;
using System.Collections.Generic;
using System.Linq;
using MNNL.Compute;

namespace MNNL.Node
{
    public class Perceptron : INode
    {
        public List<IConnection> InConnections { get; private set; }
        public List<IConnection> OutConnections { get; private set; }
        public double Input { get; private set; }
        public double Output { get; private set; }
        public double Gradient { get; private set; }
        public Func<double, double> ComputeFunc { get; private set; }
        public Func<double, double> GradientFunc { get; private set; }
        
        public Perceptron(IActivation activationFunction = null)
        {
            var ActivationFunction = activationFunction ?? new TanhSigmoidActivation();

            ComputeFunc = ActivationFunction.ComputeFunc;
            GradientFunc = ActivationFunction.GradientFunc;
            InConnections = new List<IConnection>();
            OutConnections = new List<IConnection>();
        }

        public double Forward(double? value = null)
        {
            Input = value ?? InConnections.Sum(i => i.Forward());
            Output = ComputeFunc(Input);
            return Output;
        }

        public double Backward(double? value = null)
        {
            var sumGrad = value ?? OutConnections.Sum(c => c.Backward());

            Gradient = sumGrad * GradientFunc(Input);
            return Gradient;
        }

        public override string ToString()
        {
            return string.Format("Input: {0}, Output : {1}", Input.ToString("F2"), 
                Output.ToString("F2"));
        }
    }
}
