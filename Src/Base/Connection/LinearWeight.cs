using System;
using MNNL.Parameter;

namespace MNNL.Connection
{
    public class LinearWeight : IConnection
    {
        private static readonly Random rand = new Random();
        public Func<double, double> ComputeFunc { get; private set; }
        public Func<double, double> GradientFunc { get; private set; }
        public INode InNode { get; private set; }
        public INode OutNode { get; private set; }
        public IParameter[] Parameters { get; private set; }

        public double this[int index]
        {
            get { return Parameters[index].Value; }
            set { Parameters[index].Value = value; }
        }
        
        public void Reset()
        {
            foreach (var parameter in Parameters)
                parameter.Value = rand.NextDouble();
        }

        public LinearWeight(INode inNode, INode outNode, double? initialValue = null, string Name = "w")
        {
            InNode = inNode;
            OutNode = outNode;
            Parameters = new IParameter[] {new Coefficient(Name)};
            Reset();
            ComputeFunc = d => Parameters[0].Value*d;
            GradientFunc = d => Parameters[0].Value;
            inNode.OutConnections.Add(this);
            outNode.InConnections.Add(this);

            if (initialValue != null) Parameters[0].Value = initialValue.Value;
        }

        public static LinearWeight Connect(INode inNode, INode outNode,
            double? initialValue = null, string Name = "w")
        {
            return new LinearWeight(inNode, outNode, initialValue, Name);
        }

        public double Forward(double? value = null)
        {
            return ComputeFunc(value ?? InNode.Output);
        }

        public double Backward(double? value = null)
        {
            return (value ?? OutNode.Gradient) * GradientFunc(InNode.Output);;
        }

        public double GradientOfParameter(int parameterNo = 0)
        {
            return InNode.Output*OutNode.Gradient;
        }

        public override string ToString()
        {
            return string.Format("Weight : {0}, : {1}, Output : {2}",
                this[0].ToString("F2"), InNode.Output.ToString("F2"), Forward().ToString("F2"));
        }
    }
}
