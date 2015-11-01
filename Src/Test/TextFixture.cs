using System;
using System.Collections.Generic;
using MNNL;
using MNNL.Compute;
using MNNL.Connection;
using MNNL.Network;
using MNNL.Node;
using MNNL.Train;
using MNNL.Train.Solver;

namespace Test
{
    public static class TextFixture
    {
        public static bool Test1()
        {
            var i1 = new InputNode();
            var i2 = new InputNode();
            var h1 = new Perceptron(new Linear());
            var h2 = new Perceptron(new Linear());
            var h3 = new Perceptron(new Linear());
            LinearWeight.Connect(i1, h1, 1.0);
            LinearWeight.Connect(i1, h2, 1.0);
            LinearWeight.Connect(i2, h2, 1.0);
            LinearWeight.Connect(i2, h3, 1.0);
            var ib = new BiasNode();
            LinearWeight.Connect(ib, h1, 1.0);
            LinearWeight.Connect(ib, h2, 2.0);
            LinearWeight.Connect(ib, h3, 1.0);
            var o = new Perceptron(new Linear());
            LinearWeight.Connect(h1, o, 1.0);
            LinearWeight.Connect(h2, o, -2.0);
            LinearWeight.Connect(h3, o, 1.0);
            var hb = new BiasNode();
            var whbo = new LinearWeight(hb, o);
            whbo[0] = 1.0;

            var network = new GeneralNetwork(new List<InputNode> {i1, i2}, new List<INode>{o});
            network.Forward(new[] { 0.0, 0.0 });
            network.Forward(new[] { 1.0, 0.0 });
            network.Forward(new[] { 0.0, 1.0 });
            network.Forward(new[] { 1.0, 1.0 });
            return true;

        }
        
        public static bool Test2()
        {
            var i1 = new InputNode();
            var i2 = new InputNode();
            var h1 = new Perceptron(new Heaviside(1.0));
            var h2 = new Perceptron(new Heaviside(2.0));
            var h3 = new Perceptron(new Heaviside(1.0));
            LinearWeight.Connect(i1, h1, 1.0);
            LinearWeight.Connect(i1, h2, 1.0);
            LinearWeight.Connect(i2, h2, 1.0);
            LinearWeight.Connect(i2, h3, 1.0);
            var o = new Perceptron(new Linear());
            LinearWeight.Connect(h1, o, 1.0);
            LinearWeight.Connect(h2, o, -2.0);
            LinearWeight.Connect(h3, o, 1.0);

            var network = new GeneralNetwork(new List<InputNode> { i1, i2 }, new List<INode> { o });
            network.Forward(new[] { 0.0, 0.0 });
            network.Forward(new[] { 1.0, 0.0 });
            network.Forward(new[] { 0.0, 1.0 });
            network.Forward(new[] { 1.0, 1.0 });
            return true;
        }
        
        public static bool Test3()
        {
            var mlp = new MLP(2,2,1);
            mlp[0, 0, 0][0] = 1;
            mlp[0, 0, 1][0] = 2;
            mlp[0, 1, 0][0] = 3;
            mlp[0, 1, 1][0] = 4;
            mlp[0, 2, 0][0] = 5; //weight of bias to neuron #1 connection in next layer
            mlp[0, 2, 1][0] = 6; //bias to neuron #2 in next layer
            mlp[1, 0, 0][0] = 7;
            mlp[1, 1, 0][0] = 8;
            mlp[1, 2, 0][0] = 9; //bias to output
            return true;
        }

        public static bool Test4()
        {
            var xorinput = new[] {new[]{0.0, 0.0}, new[]{1.0, 0.0}, new[]{0.0, 1.0}, new[]{1.0, 1.0}};
            var xoroutput = new[] { new[] { 0.0 }, new[] { 1.0 }, new[] { 1.0 }, new[] { 0.0 } };
            var mlp = new MLP(2, 4, 1);
            var stsp = new StandardTrainingSetProvider(xorinput, xoroutput);
            stsp.Split();
            var gdbp = new BackPropagation(mlp, stsp);

            ((GD)gdbp.Solver).Criterion = LearningCriterion.CrossEntropy;
            //((GD)gdbp.Solver).LearningRate = .01;
            ((GD)gdbp.Solver).AdaGrad = true;

            var minerr = double.MaxValue;

            gdbp.ReportReady += optimizer =>
            {
                Console.WriteLine("Epoch = {0}, Error = {1}", optimizer.CurrentEpoch, 
                    optimizer.TrainingSetProvider.TrainError);
                minerr = Math.Min(minerr, optimizer.Solver.Error);
                
                if (optimizer.Done)
                {
                    Console.ReadLine();
                }
            };

            gdbp.RunAsync().Wait();
            return true;
        }
    }
}
