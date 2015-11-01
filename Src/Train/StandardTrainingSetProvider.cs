using System;
using System.Collections.Generic;
using System.Linq;

namespace MNNL.Train
{
    public class StandardTrainingSetProvider : ITrainingSetProvider
    {
        private bool normalized;
        public double[][] AllInputs { get; private set; }
        public double[][] AllDesiredOuputs { get; private set; }
        public double[][] TrainInputs { get; private set; }
        public double[][] TrainDesiredOuputs { get; private set; }
        public double TrainError { get; set; }
        public double[][] TestInputs { get; private set; }
        public double[][] TestDesiredOuputs { get; private set; }
        public double TestError { get; set; }
        public double[][] ValidationInputs { get; private set; }
        public double[][] ValidationDesiredOuputs { get; private set; }
        public double ValidationError { get; set; }

        private double[] _InputMin;
        public double[] InputMin 
        {
            get { return _InputMin; }
            set
            {
                if(normalized) DeNormalize();

                _InputMin = value;
            }
        }

        private double[] _InputMax;
        public double[] InputMax
        {
            get { return _InputMax; }
            set
            {
                if (normalized) DeNormalize();

                _InputMax = value;
            }
        }

        private double[] _OutputMin;
        public double[] OutputMin
        {
            get { return _OutputMin; }
            set
            {
                if (normalized) DeNormalize();

                _OutputMin = value;
            }
        }

        private double[] _OutputMax;
        public double[] OutputMax
        {
            get { return _OutputMax; }
            set
            {
                if (normalized) DeNormalize();

                _OutputMax = value;
            }
        }

        public StandardTrainingSetProvider(double[][] inputs, double[][] outputs)
        {
            if(inputs.Length == 0 || inputs.Length != outputs.Length || 
                inputs[0].Length == 0 || outputs[0].Length == 0)
                throw new ArgumentException("Zero vector size or Input/Output sizes does not match.");

            AllInputs = inputs;
            AllDesiredOuputs = outputs;
            _InputMin = new double[inputs[0].Length];
            _InputMax = new double[inputs[0].Length];
            _OutputMin = new double[outputs[0].Length];
            _OutputMax = new double[outputs[0].Length];

            for (var i = 0; i < _InputMin.Length; i++)
            {
                _InputMin[i] = inputs[0][i];
                _InputMax[i] = inputs[0][i];
            }

            for (var i = 0; i < _OutputMin.Length; i++)
            {
                _OutputMin[i] = outputs[0][i];
                _OutputMax[i] = outputs[0][i];
            }

            for (var i = 1; i < inputs.Length; i++)
            {
                for (var j = 0; j < _InputMin.Length; j++)
                {
                    if (inputs[i][j] > _InputMax[j]) _InputMax[j] = inputs[i][j];
                    else if(inputs[i][j] < _InputMin[j]) _InputMin[j] = inputs[i][j];
                }
            }

            for (var i = 1; i < outputs.Length; i++)
            {
                for (var j = 0; j < _OutputMin.Length; j++)
                {
                    if (outputs[i][j] > _OutputMax[j]) _OutputMax[j] = outputs[i][j];
                    else if (outputs[i][j] < _OutputMin[j]) _OutputMin[j] = outputs[i][j];
                }
            }

            normalized = false;
        }

        public void Split(double trainRatio = 1.0, double testRatio = 0.0)
        {
            var validationRatio = 0.0;
            trainRatio = Math.Round(trainRatio * 100) / 100.0;
            testRatio = Math.Round(testRatio * 100) / 100.0;

            if (trainRatio + testRatio > 1)
            {
                var sum = trainRatio + testRatio;
                trainRatio = Math.Round(trainRatio / sum * 100) / 100.0;
                testRatio = Math.Round(testRatio / sum * 100) / 100.0;
            }

            if (trainRatio + testRatio <= .99)
            {
                validationRatio = 1.0 - (trainRatio + testRatio);
            }

            var totalCount = AllInputs.Length;
            var r = new Random();
            List<int> testSetIndexes = new List<int>(), 
                trainSetIndexes = new List<int>(), 
                validationSetIndexes = new List<int>();

            while (totalCount-- > 0)
            {
                var rnd = r.NextDouble();

                if(rnd < validationRatio) validationSetIndexes.Add(totalCount);
                else if(rnd < validationRatio + testRatio) testSetIndexes.Add(totalCount);
                else trainSetIndexes.Add(totalCount);
            }

            TrainInputs = trainSetIndexes.Select(i => AllInputs[i]).ToArray();
            TestInputs = testSetIndexes.Select(i => AllInputs[i]).ToArray();
            ValidationInputs = validationSetIndexes.Select(i => AllInputs[i]).ToArray();
            TrainDesiredOuputs = trainSetIndexes.Select(i => AllDesiredOuputs[i]).ToArray();
            TestDesiredOuputs = testSetIndexes.Select(i => AllDesiredOuputs[i]).ToArray();
            ValidationDesiredOuputs = validationSetIndexes.Select(i => AllDesiredOuputs[i]).ToArray();
        }

        public void Normalize()
        {
            if(normalized) return;
            
            normalized = true;
        }

        public void DeNormalize()
        {
            if(!normalized) return;

            normalized = false;
        }

        public void Randomize()
        {
            throw new NotImplementedException();
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
