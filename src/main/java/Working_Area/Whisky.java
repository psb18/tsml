package Working_Area;


import evaluation.storage.ClassifierResults;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.*;
import weka.core.Instance;
import weka.core.Instances;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.Random;

public class Whisky
{
    public static void taskOne() throws Exception {
        Instances train = WekaTools.loadClassificationData("C:/Users/sscar/arffs/JW_RedVsBlack/JW_RedVsBlack0_TRAIN.arff");
        Instances test = WekaTools.loadClassificationData("C:/Users/sscar/arffs/JW_RedVsBlack/JW_RedVsBlack0_TEST.arff");

        NaiveBayes naiveBayes = new NaiveBayes();
        naiveBayes.buildClassifier(train);

        File file = new File("TestFolds","testFold1.csv");
        file.getParentFile().mkdirs();

        BufferedWriter br = new BufferedWriter(new FileWriter(file));

        StringBuilder sb = new StringBuilder("JW_RedVsBlack,NaiveBayes\nNo Parameter Info\n" +
                WekaTools.accuracy(naiveBayes, test) + "\n");

        for (Instance instance : test)
        {
            int predicted = (int)naiveBayes.classifyInstance(instance);
            int actual = (int)instance.classValue();
            double[] distribution = naiveBayes.distributionForInstance(instance);

            sb.append(actual).append(",").append(predicted).append(",");

            for (double value : distribution)
            {
                sb.append(",").append(value);
            }

            sb.append("\n");
        }

        br.write(sb.toString());
        br.close();
    }

    public static ClassifierResults generateResults(Classifier c, String name) throws Exception
    {
        Instances train = WekaTools.loadClassificationData("C:/Users/sscar/arffs/JW_RedVsBlack/JW_RedVsBlack0_TRAIN.arff");
        Instances test = WekaTools.loadClassificationData("C:/Users/sscar/arffs/JW_RedVsBlack/JW_RedVsBlack0_TEST.arff");

        c.buildClassifier(train);

        File file = new File("TestFolds_" + name,"testFold1.csv");
        file.getParentFile().mkdirs();

        BufferedWriter br = new BufferedWriter(new FileWriter(file));

        StringBuilder sb = new StringBuilder("JW_RedVsBlack," + name + "\nNo Parameter Info\n" +
                WekaTools.accuracy(c, test) + "\n");

        for (Instance instance : test)
        {
            int predicted = (int)c.classifyInstance(instance);
            int actual = (int)instance.classValue();
            double[] distribution = c.distributionForInstance(instance);

            sb.append(actual).append(",").append(predicted).append(",");

            for (double value : distribution)
            {
                sb.append(",").append(value);
            }

            sb.append("\n");
        }

        br.write(sb.toString());
        br.close();

        ClassifierResults classifierResults = new ClassifierResults();
        classifierResults.loadResultsFromFile("C:/Users/sscar/tsml/TestFolds_" + name + "/testFold1.csv");
        classifierResults.findAllStats();

        return classifierResults;
    }

    public static void main(String[] args) throws Exception
    {
        //taskOne();

        System.out.println("NAIVE BAYES");
        ClassifierResults classifierResults = new ClassifierResults();
        classifierResults.loadResultsFromFile("C:/Users/sscar/tsml/TestFolds/testFold1.csv");
        classifierResults.findAllStats();
        System.out.println(classifierResults.getAcc());
        System.out.println(classifierResults.balancedAcc);
        System.out.println(classifierResults.nll);
        System.out.println(classifierResults.meanAUROC);
        System.out.println("\n");

/*        System.out.println("AODE");
        classifierResults = generateResults(new AODE(), "AODE");
        System.out.println(classifierResults.getAcc());
        System.out.println(classifierResults.balancedAcc);
        System.out.println(classifierResults.nll);
        System.out.println(classifierResults.meanAUROC);
        System.out.println("\n");*/

/*        System.out.println("AODEsr");
        classifierResults = generateResults(new AODEsr(), "AODEsr");
        System.out.println(classifierResults.getAcc());
        System.out.println(classifierResults.balancedAcc);
        System.out.println(classifierResults.nll);
        System.out.println(classifierResults.meanAUROC);
        System.out.println("\n");*/

        System.out.println("BAYESNET");
        classifierResults = generateResults(new BayesNet(), "BayesNet");
        System.out.println(classifierResults.getAcc());
        System.out.println(classifierResults.balancedAcc);
        System.out.println(classifierResults.nll);
        System.out.println(classifierResults.meanAUROC);
        System.out.println("\n");

        System.out.println("BAYESIANLOGISTICREGRESSION");
        classifierResults = generateResults(new BayesianLogisticRegression(), "BayesianLogisticRegression");
        System.out.println(classifierResults.getAcc());
        System.out.println(classifierResults.balancedAcc);
        System.out.println(classifierResults.nll);
        System.out.println(classifierResults.meanAUROC);
        System.out.println("\n");

        System.out.println("DMNBtext");
        classifierResults = generateResults(new DMNBtext(), "DMNBtext");
        System.out.println(classifierResults.getAcc());
        System.out.println(classifierResults.balancedAcc);
        System.out.println(classifierResults.nll);
        System.out.println(classifierResults.meanAUROC);
        System.out.println("\n");

        System.out.println("NAIVEBAYESSIMPLE");
        classifierResults = generateResults(new NaiveBayesSimple(), "NaiveBayesSimple");
        System.out.println(classifierResults.getAcc());
        System.out.println(classifierResults.balancedAcc);
        System.out.println(classifierResults.nll);
        System.out.println(classifierResults.meanAUROC);
        System.out.println("\n");

        System.out.println("WAODE");
        classifierResults = generateResults(new WAODE(), "WAODE");
        System.out.println(classifierResults.getAcc());
        System.out.println(classifierResults.balancedAcc);
        System.out.println(classifierResults.nll);
        System.out.println(classifierResults.meanAUROC);
    }
}
