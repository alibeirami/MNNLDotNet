namespace MNNL.Node
{
    public class BiasNode : InputNode
    {
        private readonly double bias;

        public BiasNode(double bias = 1.0)
        {
            this.bias = bias;
        }

        public override double Forward(double? value = null)
        {
            Output = bias;
            return Output;
        }

        public override string ToString()
        {
            return string.Format("Bias : {0}", Output.ToString("F2"));
        }
    }
}
