using System;
using System.Collections.Generic;
using MNNL.Node;

namespace MNNL.NodeLayer
{
    public class PerceptronLayer : INodeLayer
    {
        private readonly int NumberOfInputs;
        private readonly bool HasBiasTerm;

        public double[] Input { get; private set; }
        public double[] Output { get; private set; }
        public double[] Gradient { get; private set; }
        public List<INode> Nodes { get; private set; }
        
        public PerceptronLayer(int NumberOfNodes, IActivation activationFunction = null,
            bool HasBiasTerm = true)
        {
            NumberOfInputs = NumberOfNodes;
            Nodes = new List<INode>();
            this.HasBiasTerm = HasBiasTerm;

            for (var i = 0; i < NumberOfNodes; i++)
                Nodes.Add(new Perceptron(activationFunction));

            if (HasBiasTerm)
            {
                Input = new double[NumberOfNodes + 1];
                Output = new double[NumberOfNodes + 1];
                Gradient = new double[NumberOfNodes + 1];
                Nodes.Add(new BiasNode());
            }
            else
            {
                Output = new double[NumberOfNodes];
                Input = new double[NumberOfNodes];
                Gradient = new double[NumberOfNodes];
            }
        }
        public double[] Forward(double[] values = null)
        {
            if (values != null && values.Length != NumberOfInputs)
                throw new ArgumentException("Dimension of values does not match with Number Of Inputs.");

            for (var i = 0; i < Nodes.Count; i++)
            {
                var n = Nodes[i];

                if (values == null)
                {
                    n.Forward();
                    Input[i] = n.Input;
                }
                else
                {
                    n.Forward(values[i]);
                    Input[i] = values[i];
                }

                Output[i] = n.Output;
            }

            if (HasBiasTerm) Nodes[NumberOfInputs].Forward();

            return Output;
        }

        public double[] Backward(double[] values = null)
        {
            if (values != null && values.Length != NumberOfInputs)
                throw new ArgumentException("Dimension of values does not match with Number Of Inputs.");

            for (var i = 0; i < Nodes.Count; i++)
            {
                var n = Nodes[i];

                if (values == null)
                {
                    Gradient[i] = n.Backward();
                }
                else
                {
                    Gradient[i] = n.Backward(values[i]);
                }
            }

            return Gradient;
        }
    }
}
