package Working_Area;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class Lab4Classifier implements Classifier
{

    private final Classifier[] m_Classifiers = new J48[500];

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
        for (int i = 0; i < 500; i++)
        {
            m_Classifiers[i] = new J48();
        }

        for (Classifier c : m_Classifiers)
        {
            Instances[] split = WekaTools.splitData(data, 0.5);
            c.buildClassifier(split[0]);
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
        double[] candidates = new double[500];

        for (int i = 0; i < 500; i++)
        {
            candidates[i] = m_Classifiers[i].classifyInstance(instance);
        }

        int count = 0;
        double candidate = -1;

        // Finding majority candidate
        for (int index = 0; index < 500; index++)
        {
            if (count == 0)
            {
                candidate = candidates[index];
                count = 1;
            }
            else
            {
                if (candidates[index] == candidate)
                    count++;
                else
                    count--;
            }
        }
        return candidate;
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
    public double[] distributionForInstance(Instance instance) throws Exception {
        return new double[0];
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
