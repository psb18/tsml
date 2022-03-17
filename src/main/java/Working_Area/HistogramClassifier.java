package Working_Area;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class HistogramClassifier implements Classifier
{

    private int bins = 10;
    private int attributeIndex = 0;

    private int[][] histograms;
    private double interval;
    private double smallest;

    public void setBins(int bins)
    {
        this.bins = bins;
    }

    public void setAttributeIndex(int attributeIndex)
    {
        this.attributeIndex = attributeIndex;
    }

    /**
     * Generates a classifier. Must initialize all fields of the classifier
     * that are not being set via options (ie. multiple calls of buildClassifier
     * must always lead to the same result). Must not change the dataset
     * in any way.
     *
     * @param data set of instances serving as training data
     * @throws Exception if the classifier has not been
     *                   generated successfully
     */
    @Override
    public void buildClassifier(Instances data) throws Exception
    {
        this.histograms = new int[data.numClasses()][bins];

        this.smallest = data.kthSmallestValue(this.attributeIndex, 1);
        double largest = data.kthSmallestValue(this.attributeIndex, data.numInstances());

        this.interval = (largest - smallest) / this.bins;

        for (int i = 0; i < data.numInstances(); i++)
        {
            double attributeValue = data.get(i).value(this.attributeIndex);
            int classValue = (int)data.get(i).classValue();
            int bin = (int)((attributeValue-smallest)/this.interval);

            bin = bin >= bins ? bins-1 : bin;

            this.histograms[classValue][bin]++;
        }
    }

    /**
     * Classifies the given test instance. The instance has to belong to a
     * dataset when it's being classified. Note that a classifier MUST
     * implement either this or distributionForInstance().
     *
     * @param instance the instance to be classified
     * @return the predicted most likely class for the instance or
     * Utils.missingValue() if no prediction is made
     * @throws Exception if an error occurred during the prediction
     */
    @Override
    public double classifyInstance(Instance instance) throws Exception
    {
        double[] distribution = distributionForInstance(instance);

        int maxAt = 0;

        for (int i = 0; i < distribution.length; i++) {
            maxAt = distribution[i] > distribution[maxAt] ? i : maxAt;
        }

        return maxAt;
    }

    /**
     * Predicts the class memberships for a given instance. If
     * an instance is unclassified, the returned array elements
     * must be all zero. If the class is numeric, the array
     * must consist of only one element, which contains the
     * predicted value. Note that a classifier MUST implement
     * either this or classifyInstance().
     *
     * @param instance the instance to be classified
     * @return an array containing the estimated membership
     * probabilities of the test instance in each class
     * or the numeric prediction
     * @throws Exception if distribution could not be
     *                   computed successfully
     */
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception
    {
        double attributeValue = instance.value(this.attributeIndex);
        int bin = (int)((attributeValue-this.smallest)/this.interval);

        bin = bin >= bins ? bins-1 : bin;

        double[] distribution = new double[histograms.length];
        int total = 0;


        for (int i = 0; i < histograms.length; i++)
        {
            distribution[i] = histograms[i][bin];
            total += histograms[i][bin];
        }

        for (int i = 0; i < distribution.length; i++)
        {
            distribution[i] = distribution[i]/total;
        }

        return distribution;
    }

    /**
     * Returns the Capabilities of this classifier. Maximally permissive
     * capabilities are allowed by default. Derived classifiers should
     * override this method and first disable all capabilities and then
     * enable just those capabilities that make sense for the scheme.
     *
     * @return the capabilities of this object
     * @see Capabilities
     */
    @Override
    public Capabilities getCapabilities() {
        return null;
    }
}
