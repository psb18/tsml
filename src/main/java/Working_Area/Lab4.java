package Working_Area;

import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.LogitBoost;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class Lab4
{
    private static String[] paths =
            {
                    "C:/Users/sscar/arffs/UCI-CAWPE/bank/bank",
                    "C:/Users/sscar/arffs/UCI-CAWPE/blood/blood",
                    "C:/Users/sscar/arffs/UCI-CAWPE/breast-cancer-wisc-diag/breast-cancer-wisc-diag",
                    "C:/Users/sscar/arffs/UCI-CAWPE/breast-tissue/breast-tissue",
                    "C:/Users/sscar/arffs/UCI-CAWPE/cardiotocography-10clases/cardiotocography-10clases",
                    "C:/Users/sscar/arffs/UCI-CAWPE/conn-bench-sonar-mines-rocks/conn-bench-sonar-mines-rocks",
                    "C:/Users/sscar/arffs/UCI-CAWPE/conn-bench-vowel-deterding/conn-bench-vowel-deterding",
                    "C:/Users/sscar/arffs/UCI-CAWPE/ecoli/ecoli",
                    "C:/Users/sscar/arffs/UCI-CAWPE/glass/glass",
                    "C:/Users/sscar/arffs/UCI-CAWPE/hill-valley/hill-valley",
                    "C:/Users/sscar/arffs/UCI-CAWPE/image-segmentation/image-segmentation",
                    "C:/Users/sscar/arffs/UCI-CAWPE/ionosphere/ionosphere",
                    "C:/Users/sscar/arffs/UCI-CAWPE/iris/iris",
                    "C:/Users/sscar/arffs/UCI-CAWPE/libras/libras",
                    "C:/Users/sscar/arffs/UCI-CAWPE/optical/optical",
                    "C:/Users/sscar/arffs/UCI-CAWPE/ozone/ozone",
                    "C:/Users/sscar/arffs/UCI-CAWPE/page-blocks/page-blocks",
                    "C:/Users/sscar/arffs/UCI-CAWPE/parkinsons/parkinsons",
                    "C:/Users/sscar/arffs/UCI-CAWPE/planning/planning",
                    "C:/Users/sscar/arffs/UCI-CAWPE/post-operative/post-operative",
                    "C:/Users/sscar/arffs/UCI-CAWPE/ringnorm/ringnorm",
                    "C:/Users/sscar/arffs/UCI-CAWPE/seeds/seeds",
                    "C:/Users/sscar/arffs/UCI-CAWPE/spambase/spambase",
                    "C:/Users/sscar/arffs/UCI-CAWPE/statlog-landsat/statlog-landsat",
                    "C:/Users/sscar/arffs/UCI-CAWPE/statlog-vehicle/statlog-vehicle",
                    "C:/Users/sscar/arffs/UCI-CAWPE/steel-plates/steel-plates",
                    "C:/Users/sscar/arffs/UCI-CAWPE/synthetic-control/synthetic-control",
                    "C:/Users/sscar/arffs/UCI-CAWPE/twonorm/twonorm",
                    "C:/Users/sscar/arffs/UCI-CAWPE/vertebral-column-3clases/vertebral-column-3clases",
                    "C:/Users/sscar/arffs/UCI-CAWPE/wall-following/wall-following",
                    "C:/Users/sscar/arffs/UCI-CAWPE/waveform-noise/waveform-noise",
                    "C:/Users/sscar/arffs/UCI-CAWPE/wine-quality-white/wine-quality-white",
                    "C:/Users/sscar/arffs/UCI-CAWPE/yeast/yeast"
            };

    public static void taskOne() throws Exception
    {
        Bagging bagging = new Bagging();
        RandomForest randomForest = new RandomForest();

        StringBuilder sb = new StringBuilder();
        BufferedWriter br = new BufferedWriter(new FileWriter("bagging_forest_acc.csv"));

        sb.append("Dataset,Bagging_Accuracy,Random_Forest_Accuracy,\n");

        for (String path : paths)
        {
            Instances train = WekaTools.loadClassificationData(path + "_TRAIN.arff");
            Instances test = WekaTools.loadClassificationData(path + "_TEST.arff");

            double accuracyBagging = WekaTools.trainAndGetAccuracy(bagging, train, test);
            double accuracyForest = WekaTools.trainAndGetAccuracy(randomForest, train, test);

            sb.append(path).append(",").append(accuracyBagging).append(",").append(accuracyForest).append(",\n");
        }

        br.write(sb.toString());
        br.close();
    }

    public static void taskTwo() throws Exception {
        Bagging bagging = new Bagging();
        RandomForest randomForest = new RandomForest();

        StringBuilder sb = new StringBuilder();
        BufferedWriter br = new BufferedWriter(new FileWriter("bagging_forest_acc_2.csv"));

        sb.append("Dataset,Bagging_Accuracy,Random_Forest_Accuracy_10,Random_Forest_Accuracy_50,Random_Forest_Accuracy_100,Random_Forest_Accuracy_200,Random_Forest_Accuracy_500,Random_Forest_Accuracy_1000,\n");

        for (String path : paths)
        {
            Instances train = WekaTools.loadClassificationData(path + "_TRAIN.arff");
            Instances test = WekaTools.loadClassificationData(path + "_TEST.arff");

            double accuracyBagging = WekaTools.trainAndGetAccuracy(bagging, train, test);

            randomForest.setNumTrees(10);
            double accuracyForest10 = WekaTools.trainAndGetAccuracy(randomForest, train, test);
            randomForest.setNumTrees(50);
            double accuracyForest50 = WekaTools.trainAndGetAccuracy(randomForest, train, test);
            randomForest.setNumTrees(100);
            double accuracyForest100 = WekaTools.trainAndGetAccuracy(randomForest, train, test);
            randomForest.setNumTrees(200);
            double accuracyForest200 = WekaTools.trainAndGetAccuracy(randomForest, train, test);
            randomForest.setNumTrees(500);
            double accuracyForest500 = WekaTools.trainAndGetAccuracy(randomForest, train, test);
            randomForest.setNumTrees(1000);
            double accuracyForest1000 = WekaTools.trainAndGetAccuracy(randomForest, train, test);

            sb.append(path).append(",").append(accuracyBagging).append(",")
                    .append(accuracyForest10).append(",").append(accuracyForest50).append(",")
                    .append(accuracyForest100).append(",").append(accuracyForest200).append(",")
                    .append(accuracyForest500).append(",").append(accuracyForest1000).append(",\n");
        }

        br.write(sb.toString());
        br.close();
    }

    public static void taskThree() throws Exception {
        J48 j48 = new J48();
        RandomForest randomForest = new RandomForest();

        StringBuilder sb = new StringBuilder();
        BufferedWriter br = new BufferedWriter(new FileWriter("j48_forest500_acc.csv"));

        sb.append("Dataset,J48_Accuracy,Random_Forest_Accuracy,,\n");

        for (String path : paths)
        {
            Instances train = WekaTools.loadClassificationData(path + "_TRAIN.arff");
            Instances test = WekaTools.loadClassificationData(path + "_TEST.arff");

            double accuracyj48 = WekaTools.trainAndGetAccuracy(j48, train, test);

            randomForest.setNumTrees(500);
            double accuracyForest = WekaTools.trainAndGetAccuracy(randomForest, train, test);

            sb.append(path).append(",").append(accuracyj48).append(",").append(accuracyForest).append(",\n");
        }

        br.write(sb.toString());
        br.close();
    }

    public static void taskFour() throws Exception
    {
        Instances train = WekaTools.loadClassificationData("C:/Users/sscar/arffs/Adiac/Adiac_TRAIN.arff");
        Instances test = WekaTools.loadClassificationData("C:/Users/sscar/arffs/Adiac/Adiac_TEST.arff");

        AdaBoostM1 adaBoostM1 = new AdaBoostM1();
        System.out.println(WekaTools.trainAndGetAccuracy(adaBoostM1, train, test));

        LogitBoost logitBoost = new LogitBoost();
        System.out.println(WekaTools.trainAndGetAccuracy(logitBoost, train, test));
    }

    public static void taskFive() throws Exception
    {
        Bagging bagging = new Bagging();
        RandomForest randomForest = new RandomForest();
        randomForest.setNumTrees(500);
        AdaBoostM1 adaBoostM1 = new AdaBoostM1();
        LogitBoost logitBoost = new LogitBoost();
        J48 j48 = new J48();
        Lab4Classifier lab4Classifier = new Lab4Classifier();

        StringBuilder sb = new StringBuilder();
        BufferedWriter br = new BufferedWriter(new FileWriter("task_5_acc.csv"));

        sb.append("Dataset,Bagging_Accuracy,Random_Forest_Accuracy,AdaBoost_Accuracy,LogitBoost_Accuracy,J48_Accuracy,Lab4Classifier_Accuracy,\n");

        for (String path : paths)
        {
            Instances train = WekaTools.loadClassificationData(path + "_TRAIN.arff");
            Instances test = WekaTools.loadClassificationData(path + "_TEST.arff");

            double accuracyBagging = WekaTools.trainAndGetAccuracy(bagging, train, test);
            double accuracyForest = WekaTools.trainAndGetAccuracy(randomForest, train, test);
            double accuracyAdaBoost = WekaTools.trainAndGetAccuracy(adaBoostM1, train, test);
            double accuracyLogitBoost = WekaTools.trainAndGetAccuracy(logitBoost, train, test);
            double accuracyJ48 = WekaTools.trainAndGetAccuracy(j48, train, test);
            double accuracyLab4Classifier = WekaTools.trainAndGetAccuracy(lab4Classifier, train, test);


            sb.append(path).append(",").append(accuracyBagging).append(",")
                    .append(accuracyForest).append(",").append(accuracyAdaBoost).append(",")
                    .append(accuracyLogitBoost).append(",").append(accuracyJ48).append(",")
                    .append(accuracyLab4Classifier).append(",\n");
        }

        br.write(sb.toString());
        br.close();
    }

    public static void main(String[] args) throws Exception
    {
        //taskOne();
        //taskTwo();
        //taskThree();
        //taskFour();
        taskFive();
    }
}
