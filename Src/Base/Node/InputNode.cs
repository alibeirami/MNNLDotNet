using System;
using System.Collections.Generic;

namespace MNNL.Node
{
    public class InputNode : INode
    {
        public List<IConnection> InConnections { get; private set; }
        public List<IConnection> OutConnections { get; private set; }
        public double Input { get; set; }
        public double Output { get; protected set; }
        public double Gradient { get; private set; }
        public Func<double, double> ComputeFunc { get; private set; }
        public Func<double, double> GradientFunc { get; private set; }

        public InputNode()
        {
            InConnections = null;
            OutConnections = new List<IConnection>();
            ComputeFunc = null;
            GradientFunc = null;
            Gradient = double.NaN;
        }

        public virtual double Forward(double? value = null)
        {
            Output = value ?? Input;
            return Output;
        }

        public double Backward(double? value = null)
        {
            return double.NaN;
        }

        public override string ToString()
        {
            return string.Format("Output : {0}", Output.ToString("F2"));
        }
    }
}
