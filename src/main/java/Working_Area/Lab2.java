package Working_Area;

import weka.classifiers.trees.J48;
import weka.classifiers.trees.j48.Distribution;
import weka.classifiers.trees.j48.EntropySplitCrit;
import weka.classifiers.trees.j48.InfoGainSplitCrit;
import weka.core.Instances;

public class Lab2
{
    public static void taskOne() throws Exception {
        Instances data = WekaTools.loadClassificationData("C:/Users/sscar/arffs/balloonsapluss.arff");
        data.setClassIndex(data.numAttributes()-1);

        J48 j48 = new J48();
        j48.buildClassifier(data);

        System.out.println(j48);

        j48.setBinarySplits(true);

        System.out.println(j48);

        j48.setBinarySplits(false);
        j48.setReducedErrorPruning(true);

        System.out.println(j48);

        j48.setBinarySplits(true);

        System.out.println(j48);

        data = WekaTools.loadClassificationData("C:/Users/sscar/arffs/wdbc.arff");
        data.setClassIndex(0);

        j48 = new J48();
        j48.buildClassifier(data);

        System.out.println(j48);

        j48.setBinarySplits(true);

        System.out.println(j48);

        j48.setBinarySplits(false);
        j48.setReducedErrorPruning(true);

        System.out.println(j48);

        j48.setBinarySplits(true);

        System.out.println(j48);
    }

    public static void taskTwo() throws Exception {
        Instances data = WekaTools.loadClassificationData("C:/Users/sscar/arffs/Golf.arff");
        Distribution distribution = new Distribution(data);

        InfoGainSplitCrit infoGainSplitCrit = new InfoGainSplitCrit();
        EntropySplitCrit entropySplitCrit = new EntropySplitCrit();

        System.out.println(infoGainSplitCrit.splitCritValue(distribution));
        System.out.println(entropySplitCrit.splitCritValue(distribution));
    }

    public static void main(String[] args) throws Exception {
        //taskOne();
        taskTwo();
    }
}
