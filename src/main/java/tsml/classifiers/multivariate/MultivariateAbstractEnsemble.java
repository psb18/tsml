package tsml.classifiers.multivariate;

import evaluation.evaluators.CrossValidationEvaluator;
import evaluation.evaluators.Evaluator;
import evaluation.storage.ClassifierResults;
import machine_learning.classifiers.ensembles.AbstractEnsemble;
import machine_learning.classifiers.ensembles.voting.MajorityConfidence;
import machine_learning.classifiers.ensembles.weightings.TrainAcc;
import machine_learning.classifiers.kNN;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.interval_based.TSF;
import utilities.ErrorReport;
import utilities.ThreadingUtilities;
import utilities.multivariate_tools.MultivariateInstanceTools;
import weka.classifiers.Classifier;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.trees.J48;
import weka.core.EuclideanDistance;
import weka.core.Instances;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;

public abstract class MultivariateAbstractEnsemble extends AbstractEnsemble {


    public MultivariateAbstractEnsemble(){ }

    @Override
    public void setupDefaultEnsembleSettings() { }

    protected void initialiseModules(Instances[] data, int numClasses) throws Exception {
        //currently will only have file reading ON or OFF (not load some files, train the rest)
        //having that creates many, many, many annoying issues, especially when classifying test cases
        if (readIndividualsResults) {
            if (!resultsFilesParametersInitialised)
                throw new Exception("Trying to load "+ensembleName+" modules from file, but parameters for results file reading have not been initialised");
            loadModules(); //will throw exception if a module cannot be loaded (rather than e.g training that individual instead)
        }
        else
            trainModules(data);

        for (EnsembleModule module : modules) {
            //in case train results didnt have probability distributions, hack for old hive cote results tony todo clean
            module.trainResults.setNumClasses(numClasses);
            if (fillMissingDistsWithOneHotVectors)
                module.trainResults.populateMissingDists();

            module.trainResults.findAllStatsOnce();
        }
    }

    protected synchronized void trainModules(Instances[] data) throws Exception {

        //define the operations to build and evaluate each module, as a function
        //that will build the classifier and return train results for it, either
        //generated by the classifier itself or the trainEstimator
        List<Callable<ClassifierResults>> moduleBuilds = new ArrayList<>();
        int i=0;
        for (EnsembleModule module : modules) {
            final Classifier classifier = module.getClassifier();
            final Evaluator eval = trainEstimator.cloneEvaluator();
            final int fi = i;
            Callable<ClassifierResults> moduleBuild = () -> {
                ClassifierResults trainResults = null;

                if (EnhancedAbstractClassifier.classifierIsEstimatingOwnPerformance(classifier)) {
                    classifier.buildClassifier(data[fi]);
                    trainResults = ((EnhancedAbstractClassifier)classifier).getTrainResults();
                }
                else {
                    trainResults = eval.evaluate(classifier, data[fi]);
                    classifier.buildClassifier(data[fi]);
                }

                return trainResults;
            };

            moduleBuilds.add(moduleBuild);
            i++;
        }


        //complete the operations, either threaded via the executor service or
        //locally/sequentially
        List<ClassifierResults> results = new ArrayList<>();
        if (multiThread) {
            ExecutorService executor = ThreadingUtilities.buildExecutorService(numThreads);
            boolean shutdownAfter = true;

            results = ThreadingUtilities.computeAll(executor, moduleBuilds, shutdownAfter);
        }
        else {
            for (Callable<ClassifierResults> moduleBuild : moduleBuilds)
                results.add(moduleBuild.call());
        }


        //gather back the train results, write them if needed
        for (i = 0; i < modules.length; i++) {
            modules[i].trainResults = results.get(i);

            if (writeIndividualsResults) { //if we're doing trainFold# file writing
                String params = modules[i].getParameters();
                if (modules[i].getClassifier() instanceof EnhancedAbstractClassifier)
                    params = ((EnhancedAbstractClassifier)modules[i].getClassifier()).getParameters();
                writeResultsFile(modules[i].getModuleName(), params, modules[i].trainResults, "train"); //write results out
            }
        }
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        printlnDebug("**MTSC ENSEMBLE TRAIN: " + ensembleName + "**");

        Instances[] splitData = MultivariateInstanceTools.splitMultivariateInstances(data);

        long startTime = System.nanoTime();

        //transform data if specified
        if(this.transform!=null){
            printlnDebug(" Transform is being used: Transform = "+transform.getClass().getSimpleName());

            //  this.trainInsts = transform.process(data);
            for (int i=0;i<splitData.length;i++)
                splitData[i] = transform.process(splitData[i]);

            printlnDebug(" Transform "+transform.getClass().getSimpleName()+" complete");
            printlnDebug(" Transform "+transform.toString());
        }

        this.setupMultivariateEnsembleSettings(splitData.length);


        //housekeeping
        if (resultsFilesParametersInitialised) {
            if (readResultsFilesDirectories.length > 1)
                if (readResultsFilesDirectories.length != modules.length)
                    throw new Exception("Ensemble, " + this.getClass().getSimpleName() + ".buildClassifier: "
                            + "more than one results path given, but number given does not align with the number of classifiers/modules.");

            if (writeResultsFilesDirectory == null)
                writeResultsFilesDirectory = readResultsFilesDirectories[0];
        }

        // can classifier handle the data?
        for (int i=0;i<splitData.length;i++)
            getCapabilities().testWithFail(splitData[i]);



        //init
        this.numTrainInsts = data.numInstances();
        this.numClasses = data.numClasses();
        this.numAttributes = data.numAttributes();

        //set up modules
        initialiseModules(splitData,data.numClasses());

        //if modules' results are being read in from file, ignore the i/o overhead
        //of loading the results, we'll sum the actual buildtimes of each module as
        //reported in the files
        if (readIndividualsResults)
            startTime = System.nanoTime();

        //set up ensemble
        weightingScheme.defineWeightings(modules, numClasses);
        votingScheme.trainVotingScheme(modules, numClasses);

        buildTime = System.nanoTime() - startTime;
        if (readIndividualsResults) {
            //we need to sum the modules' reported build time as well as the weight
            //and voting definition time
            for (EnsembleModule module : modules) {
                buildTime += module.trainResults.getBuildTimeInNanos();

                //TODO see other todo in trainModules also. Currently working under
                //assumption that the estimate time is already accounted for in the build
                //time of TrainAccuracyEstimators, i.e. those classifiers that will
                //estimate their own accuracy during the normal course of training
                if (!EnhancedAbstractClassifier.classifierIsEstimatingOwnPerformance(module.getClassifier()))
                    buildTime += module.trainResults.getErrorEstimateTime();
            }
        }

        trainResults = new ClassifierResults();
        trainResults.setTimeUnit(TimeUnit.NANOSECONDS);

        if(getEstimateOwnPerformance())
            trainResults = estimateEnsemblePerformance(data); //combine modules to find overall ensemble trainpreds

        //HACK FOR CAWPE_EXTENSION PAPER:
        //since experiments expects us to make a train results object
        //and for us to record our build time, going to record it here instead of
        //editing experiments to record the buildTime at that level

        //buildTime does not include the ensemble's trainEstimator in any case, only the work required to be ready for testing
        //time unit has been set in estimateEnsemblePerformance(data);
        trainResults.turnOffZeroTimingsErrors();
        trainResults.setBuildTime(buildTime);
        trainResults.turnOnZeroTimingsErrors();

        this.testInstCounter = 0; //prep for start of testing
        this.prevTestInstance = null;

    }

    @Override
    protected void loadModules() throws Exception {
        //will look for all files and report all that are missing, instead of bailing on the first file not found
        //just helps debugging/running experiments a little
        ErrorReport errors = new ErrorReport("Errors while loading modules from file. Directories given: " + Arrays.toString(readResultsFilesDirectories));

        //for each module
        for(int m = 0; m < this.modules.length; m++){
            String readResultsFilesDirectory = readResultsFilesDirectories.length == 1 ? readResultsFilesDirectories[0] : readResultsFilesDirectories[m];

            boolean trainResultsLoaded = false;
            boolean testResultsLoaded = false;

            //try and load in the train/test results for this module
            File moduleTrainResultsFile = findResultsFile(readResultsFilesDirectory, modules[m].getModuleName(), "train", (m+1));
            if (moduleTrainResultsFile != null) {
                printlnDebug(modules[m].getModuleName() + " train loading... " + moduleTrainResultsFile.getAbsolutePath());

                modules[m].trainResults = new ClassifierResults(moduleTrainResultsFile.getAbsolutePath());
                trainResultsLoaded = true;
            }

            File moduleTestResultsFile = findResultsFile(readResultsFilesDirectory, modules[m].getModuleName(), "test", (m+1));
            if (moduleTestResultsFile != null) {
                //of course these results not actually used at all during training,
                //only loaded for future use when classifying with ensemble
                printlnDebug(modules[m].getModuleName() + " test loading..." + moduleTestResultsFile.getAbsolutePath());

                modules[m].testResults = new ClassifierResults(moduleTestResultsFile.getAbsolutePath());

                numTestInsts = modules[m].testResults.numInstances();
                testResultsLoaded = true;
            }

            if (!trainResultsLoaded)
                errors.log("\nTRAIN results files for '" + modules[m].getModuleName() + "' on '" + datasetName + "' fold '" + seed + "' not found. ");
            else if (needIndividualTrainPreds() && modules[m].trainResults.getProbabilityDistributions().isEmpty())
                errors.log("\nNo pred/distribution for instance data found in TRAIN results file for '" + modules[m].getModuleName() + "' on '" + datasetName + "' fold '" + seed + "'. ");

            if (!testResultsLoaded)
                errors.log("\nTEST results files for '" + modules[m].getModuleName() + "' on '" + datasetName + "' fold '" + seed + "' not found. ");
            else if (modules[m].testResults.numInstances()==0)
                errors.log("\nNo prediction data found in TEST results file for '" + modules[m].getModuleName() + "' on '" + datasetName + "' fold '" + seed + "'. ");
        }

        errors.throwIfErrors();
    }

    protected File findResultsFile(String readResultsFilesDirectory, String classifierName, String trainOrTest, int dimension) {
        File file = new File(readResultsFilesDirectory+classifierName+"/Predictions/"+datasetName+"Dimension"+(dimension)+"/"+trainOrTest+"Fold"+seed+".csv");
        if(!file.exists() || file.length() == 0)
            return null;
        else
            return file;
    }

    protected abstract void setupMultivariateEnsembleSettings(int instancesLength);

}
