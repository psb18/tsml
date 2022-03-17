package Working_Area;

import weka.classifiers.Classifier;
import weka.classifiers.functions.Logistic;
import weka.classifiers.lazy.IB1;
import weka.classifiers.lazy.IBk;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.J48;
import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;

import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;

public class WekaTools
{
    public static double accuracy(Classifier c, Instances test) throws Exception
    {
        int correct = 0;

        for (Instance instance : test)
        {
            double prediction = c.classifyInstance(instance);
            if (prediction == instance.classValue())
            {
                correct++;
            }
        }

        return (double)correct/(double)test.size();
    }

    public static Instances loadClassificationData(String fullPath)
    {
        Instances instances = null;

        try
        {
            FileReader reader = new FileReader(fullPath);
            instances = new Instances(reader);
        }
        catch(Exception e)
        {
            System.out.println("Exception caught: " + e);
        }

        assert instances != null;
        instances.setClassIndex(instances.numAttributes()-1);

        return instances;
    }

    public static Instances[] splitData(Instances all, double proportion)
    {
        Instances[] split = new Instances[2];

        split[0] = new Instances(all);
        split[1] = new Instances(all, 0);

        split[0].randomize(new Random());

        for (int i = 0; i < proportion*all.numInstances(); i++)
        {
            Instance instance = split[0].remove(0);
            split[1].add(instance);
        }

        return split;
    }

    public static double[] classDistribution(Instances data)
    {
        double[] distribution = new double[data.numClasses()];

        int classIndex = data.classIndex();

        AttributeStats stats = data.attributeStats(classIndex);
        int[] nominalCounts = stats.nominalCounts;

        for (int i = 0; i < nominalCounts.length; i++)
        {
            distribution[i] = (double)nominalCounts[i]/data.size();
        }

        return distribution;
    }

    public static double[][] confusionMatrix(double[] predicted, double[] actual)
    {
        HashSet<Double> uniques = new HashSet<>();
        for (double value : actual)
        {
            uniques.add(value);
        }

        double[][] confusionMatrix = new double[uniques.size()][uniques.size()];

        for (int i = 0; i < actual.length; i++)
        {
            confusionMatrix[(int)actual[i]][(int)predicted[i]]++;
        }

        return confusionMatrix;
    }

    public static double[] classifyInstances(Classifier c, Instances data) throws Exception
    {
        double[] predictions = new double[data.size()];

        for (int i = 0; i < data.size(); i++)
        {
            predictions[i] = c.classifyInstance(data.get(i));
        }

        return predictions;
    }

    public static double[] getClassValues(Instances data)
    {
        double[] values = new double[data.size()];

        for (int i = 0; i < data.size(); i++)
        {
            values[i] = data.get(i).value(data.classIndex());
        }

        return values;
    }

    public static double[][] trainAndGetConfusionMatrix(Classifier c, Instances train, Instances test) throws Exception {
        c.buildClassifier(train);

        double[] actual = getClassValues(test);
        double[] predicted = classifyInstances(c, test);

        return confusionMatrix(predicted, actual);
    }

    public static double trainAndGetAccuracy(Classifier c, Instances train, Instances test) throws Exception {
        c.buildClassifier(train);
        return accuracy(c, test);
    }

    public static void main(String[] args) throws Exception
    {
        Instances arsenal = loadClassificationData("C:/Users/sscar/arffs/Arsenal_TRAIN.arff");
        System.out.println(Arrays.toString(classDistribution(arsenal)));

        double[] actual    = { 0, 0, 1, 1, 1, 0, 0, 1, 1, 1 };
        double[] predicted = { 0, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

        System.out.println(Arrays.deepToString(confusionMatrix(predicted, actual)));

        Instances flies = loadClassificationData("C:/Users/sscar/arffs/Aedes_Female_VS_House_Fly_POWER.arff");
        //flies.setClassIndex(flies.numAttributes()-1);
        Instances[] fliesSplit = splitData(flies, 0.2);

        HistogramClassifier hc = new HistogramClassifier();
        hc.buildClassifier(fliesSplit[0]);

        double[] fliesTestActual = getClassValues(fliesSplit[1]);
        double[] fliesTestPred = classifyInstances(hc, fliesSplit[1]);

        System.out.println(Arrays.deepToString(confusionMatrix(fliesTestPred, fliesTestActual)));

        MajorityClass mc = new MajorityClass();
        mc.buildClassifier(fliesSplit[0]);

        double[] fliesPredMc = classifyInstances(mc, fliesSplit[1]);

        System.out.println(Arrays.deepToString(confusionMatrix(fliesPredMc, fliesTestActual)));

        ZeroR zeroR = new ZeroR();
        zeroR.buildClassifier(fliesSplit[0]);

        double[] fliesPredZr = classifyInstances(zeroR, fliesSplit[1]);

        System.out.println(Arrays.deepToString(confusionMatrix(fliesPredZr, fliesTestActual)));

        System.out.println();
        System.out.println("-------Flies Investigation--------");
        System.out.println();

        //Classifiers
        IB1 ib1 = new IB1();
        IBk iBk = new IBk();
        J48 j48 = new J48();
        Logistic logistic = new Logistic();
        MajorityClass majorityClass = new MajorityClass();

        Instances[] split = splitData(flies, 0.3);

        System.out.println(Arrays.deepToString(trainAndGetConfusionMatrix(ib1, split[0], split[1])));
        System.out.println(Arrays.deepToString(trainAndGetConfusionMatrix(iBk, split[0], split[1])));
        System.out.println(Arrays.deepToString(trainAndGetConfusionMatrix(j48, split[0], split[1])));
        System.out.println(Arrays.deepToString(trainAndGetConfusionMatrix(logistic, split[0], split[1])));
        System.out.println(Arrays.deepToString(trainAndGetConfusionMatrix(majorityClass, split[0], split[1])));

        double[][] accuracies = new double[30][5];

        for (int i = 0; i < 30; i++)
        {
            System.out.println("Pass: " + i);
            split = splitData(flies, 0.3);
            accuracies[i][0] = trainAndGetAccuracy(ib1, split[0], split[1]);;
            accuracies[i][1] = trainAndGetAccuracy(iBk, split[0], split[1]);
            accuracies[i][2] = trainAndGetAccuracy(j48, split[0], split[1]);
            accuracies[i][3] = trainAndGetAccuracy(logistic, split[0], split[1]);
            accuracies[i][4] = trainAndGetAccuracy(majorityClass, split[0], split[1]);
        }

        BufferedWriter br = new BufferedWriter(new FileWriter("accuracies.csv"));
        StringBuilder sb = new StringBuilder();

        for (double[] accuracy : accuracies)
        {
            for (double classifier : accuracy)
            {
                sb.append(classifier);
                sb.append(",");
            }
            sb.append("\n");
        }

        br.write(sb.toString());
        br.close();
        //Saved in project root
    }
}