using System;

namespace MNNL.Compute
{
    public class Heaviside : IActivation
    {
        public Func<double, double> ComputeFunc { get; private set; }
        public Func<double, double> GradientFunc { get; private set; }

        public Heaviside(double threshold = 0.0)
        {
            ComputeFunc = d => d >= threshold ? 1 : 0.0;
            GradientFunc = d => 0;
        }
    }
}
