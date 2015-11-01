using System;
using System.Collections.Generic;
using System.Linq;
using MNNL.Node;

namespace MNNL.Network
{
    public class GeneralNetwork : INetwork
    {
        private readonly double[] _Input;
        public double[] Input
        {
            get { return _Input; }
            private set
            {
                var il = StagedNodes[0];

                if(value.Length != NumberOfInputs && il.Length != NumberOfInputs) 
                    throw new ArgumentException();

                for (var i = 0; i < NumberOfInputs; i++)
                {
                    ((InputNode)il[i]).Input = value[i];
                    _Input[i] = value[i];
                }
            }
        }
        
        public double[] Output { get; private set; }
        public double[] Gradient { get; private set; }
        public List<INode> Nodes { get; private set; }
        public int NumberOfInputs { get; private set; }
        public int NumberOfOutputs { get; private set; }
        public INode[][] StagedNodes { get; private set; }

        private readonly List<INode> outNodes; 

        public GeneralNetwork(List<InputNode> inNodes, List<INode> outNodes)
        {
            NumberOfInputs = inNodes.Count;
            _Input = new double[NumberOfInputs];
            var nodes = new List<List<INode>> {inNodes.Cast<INode>().ToList()};
            var current = nodes[0];
            var visitedNodes = new HashSet<INode>(current);
            var prevNodes = new List<INode>();
            List<INode> nextNodes;

            do
            {
                nextNodes = new List<INode>();

                var cpprev = prevNodes;
                prevNodes.AddRange(current.Where(c => c.InConnections != null).SelectMany(s => s.InConnections).
                    Select(s => s.InNode).Distinct().Where(n => !cpprev.Contains(n)));

                foreach (var onode in current.SelectMany(c => c.OutConnections).
                    Where(c => !visitedNodes.Contains(c.OutNode)).Select(c => c.OutNode))
                {
                    nextNodes.Add(onode);
                    visitedNodes.Add(onode);
                }

                prevNodes = current;
                current = nextNodes;

                if (nextNodes.Count != 0) nodes.Add(nextNodes);
            } while (nextNodes.Count != 0);

            this.outNodes = outNodes;
            NumberOfOutputs = outNodes.Count;
            Output = new double[NumberOfOutputs];
            StagedNodes = nodes.Select(n => n.ToArray()).ToArray();
            Nodes = nodes.SelectMany(s => s).ToList();
        }

        public double[] Forward(double[] values = null)
        {
            if (values != null && values.Length != NumberOfInputs)
                throw new ArgumentException("Dimension of values does not match with NumberOfInputs.");

            if (values != null) Input = values;

            foreach (var nodes in StagedNodes)
            {
                foreach (var node in nodes)
                {
                    node.Forward();
                }
            }

            for (var i = 0; i < outNodes.Count; i++)
            {
                Output[i] = outNodes[i].Output;
            }

            return Output;
        }

        public double[] Backward(double[] values)
        {
            var lastLayerNodes = StagedNodes.Last();

            for (var j = 0; j < NumberOfOutputs; j++)
            {
                if (values != null)
                {
                    lastLayerNodes[j].Backward(values[j]);
                }
                else
                {
                    lastLayerNodes[j].Backward();
                }
            }

            for (var j = StagedNodes.Length - 2; j >= 0; j--)
            {
                var layer = StagedNodes[j];

                foreach (var node in layer)
                {
                    node.Backward();
                }
            }

            return null;
        }

        public byte[] Save()
        {
            throw new NotImplementedException();
        }

        public void Load(byte[] state)
        {
            throw new NotImplementedException();
        }

    }
}
