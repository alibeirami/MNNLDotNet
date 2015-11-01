namespace MNNL.Parameter
{
    public class Coefficient : IParameter
    {
        public string Name { get; private set; }
        public double Value { get; set; }

        public Coefficient(string Name = "w")
        {
            this.Name = Name;
        }
    }
}