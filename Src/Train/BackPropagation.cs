using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using MNNL.Train.Solver;

namespace MNNL.Train
{
    public class BackPropagation : IOptimizer
    {
        private CancellationTokenSource cts;

        public int CurrentEpoch { get; private set; }
        public INetwork Network { get; private set; }
        public ITrainingSetProvider TrainingSetProvider { get; private set; }
        public ISolver Solver { get; set; }
        public bool Done { get; set; }
        public double MinError { get; set; }
        public double MaxEpoch { get; set; }
        public int BatchSize { get; set; }

        public event OptimizerReportHandler ReportReady;

        public BackPropagation(INetwork network, ITrainingSetProvider trainingSet,
            ISolver solver = null)
        {
            Network = network;
            TrainingSetProvider = trainingSet;
            CurrentEpoch = 0;
            MinError = 1e-6;
            MaxEpoch = 1e6;
            BatchSize = trainingSet.TrainInputs.Length;
            Solver = solver ?? new GD(network);
        }
        public void RunEpoch()
        {
            Solver.RunEpoch(TrainingSetProvider.TrainInputs, TrainingSetProvider.TrainDesiredOuputs);
            TrainingSetProvider.TrainError = Solver.Error;
            Done = ++CurrentEpoch >= MaxEpoch || Solver.Error < MinError;
            TrainingSetProvider.TestError = EvalError(TrainingSetProvider.TestInputs, 
                TrainingSetProvider.TestDesiredOuputs);
            TrainingSetProvider.ValidationError = EvalError(TrainingSetProvider.ValidationInputs, 
                TrainingSetProvider.ValidationDesiredOuputs);

            if (ReportReady != null) ReportReady(this);
            if (Done) return;

            //update weights
            Solver.UpdateParameters();
        }

        public double EvalError(double[][] inputs, double[][] outputs)
        {
            var sumError = new double[Network.NumberOfOutputs];

            for (var i = 0; i < inputs.Length; i++)
            {
                //forward propagation
                Network.Forward(inputs[i]);

                for (var j = 0; j < Network.NumberOfOutputs; j++)
                {
                    double noj = Network.Output[j], doij = outputs[i][j];
                    sumError[j] += Solver.LossFunc(doij, noj);
                }
            }

            return sumError.Sum(s => s / inputs.Length);
        }

        public Task RunAsync()
        {
            cts = new CancellationTokenSource();
            Done = false;

            return Task.Factory.StartNew(() =>
            {
                while (!cts.IsCancellationRequested && !Done)
                {
                    RunEpoch();
                }
            });
        }

        public void Stop()
        {
            if(cts != null) cts.Cancel();
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
