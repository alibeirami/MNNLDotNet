using System;

namespace MNNL.Compute
{
    public class Linear : IActivation
    {
        public Func<double, double> ComputeFunc { get; private set; }
        public Func<double, double> GradientFunc { get; private set; }

        public Linear()
        {
            ComputeFunc = d => d;
            GradientFunc = d => 1;
        }
    }
}
