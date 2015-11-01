using System;
using System.Collections.Generic;
using System.Linq;
using MNNL.Connection;
using MNNL.Node;

namespace MNNL.ConnectionLayer
{
    public class LinearWeightLayer : IConnectionLayer
    {
        public List<INode> InputNodes { get; private set; }
        public List<INode> OutputNodes { get; private set; }
        public List<IConnection> Connections { get; private set; }

        public IConnection this[int inputNeuronNo, int outputNeuronNo]
        {
            get 
            {
                return inputNeuronNo < InputNodes.Count && outputNeuronNo < OutputNodes.Count
                    ? InputNodes[inputNeuronNo].OutConnections[outputNeuronNo] : null;
            }
        }

        public LinearWeightLayer(INodeLayer inputLayer, INodeLayer outputLayer)
        {
            Connections = new List<IConnection>();
            InputNodes = inputLayer.Nodes;
            OutputNodes = outputLayer.Nodes;

            foreach (var inode in inputLayer.Nodes)
            {
                foreach (var onode in outputLayer.Nodes.Where(n => !(n is InputNode)))
                {
                    Connections.Add(new LinearWeight(inode, onode));
                }
            }
        }

        public double[] Forward(double[] values = null)
        {
            if (values != null && values.Length != Connections.Count)
                throw new ArgumentException("Dimension of values does not match with Connections Count.");

            var Output = new double[Connections.Count];

            for (var i = 0; i < Connections.Count; i++)
            {
                var connection = Connections[i];

                if (values == null)
                    Output[i] = connection.Forward();
                else
                    Output[i] = connection.Forward(values[i]);
            }

            return Output;
        }

        public double[] Backward(double[] values = null)
        {
            if (values != null && values.Length != Connections.Count)
                throw new ArgumentException("Dimension of values does not match with Connections Count.");

            var Gradient = new double[Connections.Count];

            for (var i = 0; i < Connections.Count; i++)
            {
                var connection = Connections[i];

                if (values == null)
                    Gradient[i] = connection.Backward();
                else
                    Gradient[i] = connection.Backward(values[i]);
            }

            return Gradient;
        }

        public override string ToString()
        {
            return string.Format("{0} Inputs, {1} Outputs which {2} of them are BiasNodes, Total {3} Connections.",
                InputNodes.Count, OutputNodes.Count, OutputNodes.Count(c => c is BiasNode), Connections.Count);
        }
    }
}
