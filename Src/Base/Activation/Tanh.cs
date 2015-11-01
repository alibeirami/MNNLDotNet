using System;

namespace MNNL.Compute
{
    public class TanhSigmoidActivation : IActivation
    {
        public Func<double, double> ComputeFunc { get; private set; }
        public Func<double, double> GradientFunc { get; private set; }

        public TanhSigmoidActivation()
        {
            ComputeFunc = Math.Tanh;
            GradientFunc = d =>
            {
                var tanh = Math.Tanh(d);
                return 1 - tanh*tanh;
            };
        }
    }
}
