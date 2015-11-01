using System;
using System.Collections.Generic;
using System.Linq;
using MNNL.Node;

namespace MNNL.NodeLayer
{
    public class InputLayer : INodeLayer
    {
        private readonly int NumberOfInputs;
        private readonly bool HasBiasTerm;

        public double[] Input { get; private set; }
        public List<INode> Nodes { get; private set; }
        public double[] Output { get; private set; }
        public double[] Gradient { get; private set; }

        public InputLayer(int NumberOfInputs, bool HasBiasTerm = true)
        {
            this.NumberOfInputs = NumberOfInputs;
            this.HasBiasTerm = HasBiasTerm;
            Nodes = new List<INode>();
         
            for (var i = 0; i < NumberOfInputs; i++)
                Nodes.Add(new InputNode());

            if (HasBiasTerm) Nodes.Add(new BiasNode());
        
            Input = new double[NumberOfInputs];
            Output = new double[NumberOfInputs];
            Gradient = new double[NumberOfInputs];
        }

        public double[] Forward(double[] values)
        {
            if(values == null) 
                throw new ArgumentException("Input can notbe null.");
            if(values.Length != NumberOfInputs)
                throw new ArgumentException("Dimension of values does not match with Number Of Inputs.");

            for (var i = 0; i < NumberOfInputs; i++)
            {
                var n = Nodes[i];

                n.Forward(values[i]);

                Input[i] = values[i];
                Output[i] = n.Output;
            }

            if (HasBiasTerm) Nodes[NumberOfInputs].Forward();

            return Output;
        }

        public double[] Backward(double[] values = null)
        {
            throw new NotImplementedException();
        }
    }
}
