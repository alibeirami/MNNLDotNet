using System;

namespace MNNL.Compute
{
    public class Sigmoid : IActivation
    {
        public Func<double, double> ComputeFunc { get; private set; }
        public Func<double, double> GradientFunc { get; private set; }

        public Sigmoid(double k = 1.0)
        {
            ComputeFunc = d => 1.0/(1 + Math.Exp(-k*d));
            GradientFunc = d =>
            {
                var fx = ComputeFunc(d);
                return k*fx*(1 - fx);
            };
        }
    }
}
