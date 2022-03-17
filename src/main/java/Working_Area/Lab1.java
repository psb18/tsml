package Working_Area;

import weka.core.Instance;
import weka.core.Instances;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;

import java.io.FileReader;
import java.util.Arrays;

public class Lab1
{
    public static Instances loadData(String filePath)
    {
        Instances train = null;

        try
        {
            FileReader reader = new FileReader(filePath);
            train = new Instances(reader);
        }
        catch(Exception e)
        {
            System.out.println("Exception caught: " + e);
        }

        return train;
    }

    public static void main(String[] args)
    {
        Instances train = loadData("C:/Users/sscar/IdeaProjects/MLLab1/src/Arsenal_TRAIN.arff");
        Instances test = loadData("C:/Users/sscar/IdeaProjects/MLLab1/src/Arsenal_TEST.arff");

        System.out.println(train.numInstances());
        System.out.println(test.numAttributes());
        System.out.println(train.attributeStats(3).nominalCounts[2]);
        System.out.println(Arrays.toString(test.instance(4).toDoubleArray()));
        System.out.println(train.toString());
        train.deleteAttributeAt(2);
        test.deleteAttributeAt(2);
        System.out.println(train.toString());
        System.out.println(test.toString());

        train = loadData("C:/Users/sscar/IdeaProjects/MLLab1/src/Arsenal_TRAIN.arff");
        test = loadData("C:/Users/sscar/IdeaProjects/MLLab1/src/Arsenal_TEST.arff");

        train.setClassIndex(train.numAttributes()-1);
        test.setClassIndex(test.numAttributes()-1);

        System.out.println();
        System.out.println("------------------------------");
        System.out.println();

        try {
            NaiveBayes naiveBayes = new NaiveBayes();
            naiveBayes.buildClassifier(train);
            IBk ibk = new IBk();
            ibk.buildClassifier(train);

            int nbCorrect = 0;
            int ibkCorrect = 0;

            for (Instance instance: test)
            {
                double correctValue = instance.toDoubleArray()[3];
                double nbPrediction = naiveBayes.classifyInstance(instance);
                double ibkPrediction = ibk.classifyInstance(instance);

                System.out.println(
                        "Actual: " + correctValue + "; Naive bayes: " + nbPrediction + "; IBk: " + ibkPrediction);

                if (correctValue == nbPrediction)
                {
                    nbCorrect++;
                }
                if (correctValue == ibkPrediction)
                {
                    ibkCorrect++;
                }
            }

            System.out.println("Naive bayes accuracy: " + (double)nbCorrect / (double)test.size());
            System.out.println("IBk accuracy: " + (double)ibkCorrect / (double)test.size());

            for (Instance instance: test)
            {
                System.out.println(instance.toString());
                System.out.println("Naive bayes distribution: " +
                        Arrays.toString(naiveBayes.distributionForInstance(instance)));
                System.out.println("IBk distribution: " + Arrays.toString(ibk.distributionForInstance(instance)));
            }
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }
    }
}
