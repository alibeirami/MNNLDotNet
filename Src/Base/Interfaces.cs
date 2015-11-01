//السلام علیک یا اباصالح المهدی
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace MNNL
{
    public interface IStateful
    {
        byte[] Save();
        void Load(byte[] state);
    }

    public interface IUnit
    {
        double Forward(double? value = null);
        double Backward(double? value = null);
    }
    
    public interface IUnitArray
    {
        double[] Forward(double[] values = null);
        double[] Backward(double[] values = null);
    }
    
    public interface IActivation
    {
        Func<double, double> ComputeFunc { get; }
        Func<double, double> GradientFunc { get; }
    }
    
    public interface IParameter
    {
        string Name { get; }
        double Value { get; set; }
    }
    
    public interface IAdjustable
    {
        IParameter[] Parameters { get; }
        double this[int index] { get; set; }
        void Reset();
        double GradientOfParameter(int parameterNo = 0);
    }
    
    public interface IConnection : IActivation, IUnit, IAdjustable
    {
        INode InNode { get; }
        INode OutNode { get; }
    }

    public interface INode : IActivation, IUnit
    {
        double Input { get; }
        double Output { get; }
        double Gradient { get; }
        List<IConnection> InConnections { get; }
        List<IConnection> OutConnections { get; }
    }

    public interface INodeLayer : IUnitArray
    {
        double[] Input { get; }
        double[] Output { get; }
        double[] Gradient { get; }
        List<INode> Nodes { get; }
    }
    
    public interface IConnectionLayer : IUnitArray
    {
        List<INode> InputNodes { get; }
        List<INode> OutputNodes { get; } 
        List<IConnection> Connections { get; }
        IConnection this[int inputNeuronNo, int outputNeuronNo] { get; }
    }
    
    public interface INetwork : INodeLayer, IStateful
    {
        int NumberOfInputs { get; }
        int NumberOfOutputs { get; }
        INode[][] StagedNodes { get; }
    }

    public interface ILayeredNetwork : INetwork
    {
        INodeLayer[] NodeLayers { get; }
        IConnection this[int inputLayerNo, int inputNeuronNo, int outputNeuronNo] { get; }
    }

    public delegate void OptimizerReportHandler(IOptimizer Optimizer);
    
    public enum LearningCriterion { MSE, CrossEntropy }
    [Flags]
    public enum LearningRegularization { None = 0, L1Norm = 1, L2Norm = 2, Dropout = 4 }

    public interface ITrainingSetProvider : IStateful
    {
        double[][] AllInputs { get; }
        double[][] AllDesiredOuputs { get; }
        double[][] TrainInputs { get; }
        double[][] TrainDesiredOuputs { get; }
        double TrainError { get; set; }
        double[][] TestInputs { get; }
        double[][] TestDesiredOuputs { get; }
        double TestError { get; set; }
        double[][] ValidationInputs { get; }
        double[][] ValidationDesiredOuputs { get; }
        double ValidationError { get; set; }
        double[] InputMin { get; set; }
        double[] InputMax { get; set; }
        double[] OutputMin { get; set; }
        double[] OutputMax { get; set; }
        void Split(double trainRatio, double testRatio);
        void Normalize();
        void DeNormalize();
        void Randomize();
    }

    public interface ISolver
    {
        LearningCriterion Criterion { get; }
        LearningRegularization Regularization { get; }
        INetwork Network { get; }
        double Error { get; }
        void RunEpoch(double[][] inputs, double[][] desiredOuputs);
        void UpdateParameters();
        double LossFunc(double desiredOutput, double estimatedOutput);
    }

    public interface IOptimizer : IStateful
    {
        ITrainingSetProvider TrainingSetProvider { get; }
        ISolver Solver { get; set; }
        bool Done { get; set; }
        double MinError { get; set; }
        double MaxEpoch { get; set; }
        int CurrentEpoch { get; }
        INetwork Network { get; }
        event OptimizerReportHandler ReportReady;
        double EvalError(double[][] inputs, double[][] outputs);
        void RunEpoch();
        Task RunAsync();
        void Stop();
    }
}
