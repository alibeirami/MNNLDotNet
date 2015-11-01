using System;

namespace MNNL.Compute
{
    public class RectifiedLinear : IActivation
    {
        public Func<double, double> ComputeFunc { get; private set; }
        public Func<double, double> GradientFunc { get; private set; }

        public RectifiedLinear(double leakSlope = 0.0)
        {
            ComputeFunc = d => d > 0 ? 1 : leakSlope;
            GradientFunc = d => d > 0 ? d : leakSlope * d;
        }
    }
}
