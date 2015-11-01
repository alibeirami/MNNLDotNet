using System;
using System.Collections.Generic;
using System.Linq;
using MNNL.ConnectionLayer;
using MNNL.NodeLayer;

namespace MNNL.Network
{
    public class MLP : ILayeredNetwork
    {
        private List<IConnectionLayer> ConnectionLayers;

        public int NumberOfInputs { get; private set; }
        public int NumberOfOutputs { get; private set; }
        
        public double[] Input 
        { 
            get { return NodeLayers[0].Input; } 
        }

        public INode[][] StagedNodes { get; private set; }
        public double[] Output { get; private set; }
        public double[] Gradient { get; private set; }
        public List<INode> Nodes { get; private set; }
        public INodeLayer[] NodeLayers { get; private set; }

        public IConnection this[int inputLayerNo, int inputNeuronNo, int outputNeuronNo]
        {
            get 
            {
                return inputLayerNo < ConnectionLayers.Count ? 
                    ConnectionLayers[inputLayerNo][inputNeuronNo, outputNeuronNo] : null;
            }
        }

        public MLP(int numberOfInputs, params Tuple<int, IActivation>[] neuronCountActivationFunctionPack)
        {
            if (neuronCountActivationFunctionPack == null || neuronCountActivationFunctionPack.Length < 1)
                throw new ArgumentException("At Least 1 output Layer should be specified.");
            //input layer
            var il = new InputLayer(numberOfInputs);
            //hidden layer with bias
            var layers = neuronCountActivationFunctionPack.Take(neuronCountActivationFunctionPack.Length - 1).
                Select(tuple => new PerceptronLayer(tuple.Item1, tuple.Item2)).
                Cast<INodeLayer>().ToList();
            //output layer without bias
            var lastLayerTuple = neuronCountActivationFunctionPack.Last();
            layers.Add(new PerceptronLayer(lastLayerTuple.Item1, lastLayerTuple.Item2, false));
            Create(il, layers.ToArray());
        }

        public MLP(IActivation activationFunction, params int[] numNeuronInLayers)
        {
            if (numNeuronInLayers == null || numNeuronInLayers.Length < 2)
                throw new ArgumentException("At Least 2 Layers should have neuron count.");
            //input layer
            var il = new InputLayer(numNeuronInLayers[0]);
            //hidden layer with bias
            var layers = numNeuronInLayers.Skip(1).Take(numNeuronInLayers.Length - 2).
                Select(numNeuronInLayer => new PerceptronLayer(numNeuronInLayer, activationFunction)).
                Cast<INodeLayer>().ToList();
            //output layer without bias
            layers.Add(new PerceptronLayer(numNeuronInLayers.Last(), activationFunction, false));
            Create(il, layers.ToArray());
        }

        public MLP(params int[] NumNeuronInLayers) : this(null, NumNeuronInLayers)
        {
        }

        public MLP(InputLayer inputLayer, INodeLayer[] layers)
        {
            Create(inputLayer, layers);
        }

        private void Create(InputLayer inputLayer, INodeLayer[] layers)
        {
            NodeLayers = new INodeLayer[layers.Length + 1];
            ConnectionLayers = new List<IConnectionLayer>();
            var nodes = new List<List<INode>>();
            NumberOfInputs = inputLayer.Input.Length;
            NumberOfOutputs = layers.Last().Output.Length;
            NodeLayers[0] = inputLayer;
            nodes.Add(inputLayer.Nodes);

            for (int i = 0; i < layers.Length; i++)
            {
                var ih = layers[i];
                NodeLayers[i+1] = ih;
                nodes.Add(ih.Nodes);
            }

            for (var i = 0; i < NodeLayers.Length - 1; i++)
            {
                var cl = new LinearWeightLayer(NodeLayers[i], NodeLayers[i + 1]);
                ConnectionLayers.Add(cl);
            }

            StagedNodes = nodes.Select(n => n.ToArray()).ToArray();
            Nodes = nodes.SelectMany(s => s).ToList();
        }

        public double[] Forward(double[] values = null)
        {
            if (values != null && values.Length != NumberOfInputs)
                throw new ArgumentException("Dimension of values does not match with NumberOfInputs.");

            NodeLayers[0].Forward(values);
            
            for (var i = 1; i < NodeLayers.Length; i++)
            {
                ConnectionLayers[i-1].Forward();
                NodeLayers[i].Forward();
            }

            Output = NodeLayers.Last().Output;
            return Output;
        }

        public double[] Backward(double[] values = null)
        {
            throw new NotImplementedException();
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
