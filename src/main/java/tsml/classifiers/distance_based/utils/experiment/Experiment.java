package tsml.classifiers.distance_based.utils.experiment;

import com.beust.jcommander.IStringConverter;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import evaluation.storage.ClassifierResults;
import experiments.Experiments;
import experiments.data.DatasetLoading;
import org.junit.Assert;
import tsml.classifiers.*;
import tsml.classifiers.distance_based.proximity.ProximityForest;
import tsml.classifiers.distance_based.proximity.ProximityForestWrapper;
import tsml.classifiers.distance_based.proximity.ProximityTree;
import tsml.classifiers.distance_based.utils.classifiers.*;
import tsml.classifiers.distance_based.utils.classifiers.results.ResultUtils;
import tsml.classifiers.distance_based.utils.strings.StrUtils;
import tsml.classifiers.distance_based.utils.system.logging.LogUtils;
import tsml.classifiers.distance_based.utils.system.logging.Loggable;
import tsml.classifiers.distance_based.utils.system.memory.MemoryWatcher;
import tsml.classifiers.distance_based.utils.system.timing.StopWatch;
import utilities.ClassifierTools;
import utilities.FileUtils;
import weka.classifiers.Classifier;
import weka.core.Randomizable;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

import static tsml.classifiers.distance_based.utils.system.SysUtils.hostName;
import static weka.core.Debug.OFF;

public class Experiment implements Copier {
    
    public Experiment() {
        setExperimentLogLevel(Level.ALL);
        addConfigs(ProximityForest.Config.values());
        addConfigs(ProximityTree.Config.values());
        Builder<ProximityForestWrapper> pfwR5 = () -> {
            ProximityForestWrapper pfw = new ProximityForestWrapper();
            pfw.setR(5);
            return pfw;
        };
        addConfig("PF_WRAPPER", pfwR5);
        addConfig("PF_WRAPPER_R5", pfwR5);
        addConfig("PF_WRAPPER_R1",  () -> {
            ProximityForestWrapper pfw = new ProximityForestWrapper();
            pfw.setR(1);
            return pfw;
        });
        addConfig("PF_WRAPPER_R10",  () -> {
            ProximityForestWrapper pfw = new ProximityForestWrapper();
            pfw.setR(10);
            return pfw;
        });
    }
    
    @Parameter(names = {"-c", "--classifier"}, description = "The classifier to use.")
    private String classifierName;

    @Parameter(names = {"-s", "--seed"}, description = "The seed used in resampling the data and producing random numbers in the classifier.")
    private Integer seed;

    @Parameter(names = {"-r", "--results"}, description = "The results directory.")
    private String resultsDirPath;

    @Parameter(names = {"-d", "--data"}, description = "The data directory.")
    private String dataDirPath;

    @Parameter(names = {"-p", "--problem"}, description = "The problem name. E.g. GunPoint.")
    private String problemName;

    @Parameter(names = {"--cp", "--checkpoint"}, description = "Periodically save the classifier to disk. Default: off")
    private boolean checkpoint = false;

    @Parameter(names = {"--rcp", "--removeCheckpoint"}, description = "Remove any checkpoints upon completion of a train time contract. The assumption here is that once the classifier has built in the given time limit, no further work will be done and the checkpoint can be safely removed. In other words, the assumption is that the checkpoint is only useful if the classifier gets stoppped mid-build and must be restarted. When the classifier finishes building, the checkpoint files are redundant, therefore. Note this does not affect multiple contracts as the checkpoint files are copied before removal. I.e. a contract of 1h completes and leaves behind some checkpoint files. These are copied over to the subsequent 2h contract before removal from the 1h contract working area. Default: off")
    private boolean removeCheckpoint = false;

    @Parameter(names = {"--cpi", "--checkpointInterval"}, description = "The minimum interval between checkpoints. Classifiers may not produce checkpoints within the checkpoint interval time from the last checkpoint. Therefore, classifiers may produce checkpoints with (much) larger intervals inbetween, depending on their checkpoint frequency. Default: 1h")
    private String checkpointInterval = "1h";

    @Parameter(names = {"--scp", "--snapshotCheckpoints"}, description = "Keep every checkpoint as a snapshot of the classifier model at that point in time. When disabled, classifiers overwrite their last checkpoint. When enabled, classifiers will write checkpoints with a time stamp rather than overwriting previous checkpoints. Default: off")
    private boolean snapshotCheckpoints = false;

    @Parameter(names = {"--ttl", "--trainTimeLimit"}, description = "Contract the classifier to build in a set time period. Give this option two arguments in the form of '--contractTrain <amount> <units>', e.g. '--contractTrain 5 minutes'")
    private List<String> trainTimeLimitStrs = new ArrayList<>();
    private List<TimeSpan> trainTimeLimits = new ArrayList<>();

    @Parameter(names = {"-e", "--evaluate"}, description = "Estimate the train error. Default: false")
    private boolean evaluateClassifier = false;

    @Parameter(names = {"-t", "--threads"}, description = "The number of threads to use. Set to 0 or less for all available processors at runtime. Default: 1")
    private int numThreads = 1;

    @Parameter(names = {"--trainOnly"}, description = "Only train the classifier, do not test.")
    private boolean trainOnly = false;

    @Parameter(names = {"-o", "--overwrite"}, description = "Overwrite previous results.")
    private boolean overwriteResults = false;

    public Level getExperimentLogLevel() {
        return log.getLevel();
    }

    public void setExperimentLogLevel(final Level level) {
        log.setLevel(level);
    }

    private static class LogLevelConverter implements IStringConverter<Level> {

        @Override public Level convert(final String s) {
            return Level.parse(s.toUpperCase());
        }
    }

    @Parameter(names = {"-l", "--logLevel"}, description = "The amount of logging. This should be set to a Java log level. Default: OFF", converter = LogLevelConverter.class)
    private Level logLevel = Level.OFF;

    private final Logger log = LogUtils.buildLogger(this);
    
    private void addConfig(String key, Builder<? extends Classifier> value) {
        classifierLookup.put(key, value);
    }
    
    private <A> void addConfigs(Iterable<A> entries) {
        for(A entry : entries) {
            if(!(entry instanceof Enum)) {
                throw new IllegalArgumentException("not an enum entry");
            }
            String key = ((Enum) entry).name();
            Builder<? extends Classifier> value;
            try {
                value = (Builder<? extends Classifier>) entry;
            } catch(ClassCastException e) {
                throw new IllegalArgumentException("not an instance of Configurer which accepts a Classifier instance");
            }
            addConfig(key, value);
        }
    }
    
    private <A> void addConfigs(A... entries) {
        addConfigs(Arrays.asList(entries));
    }
    
    private static <A, B> Map<String, Configurer<? extends Classifier>> enumToMap(Class clazz) {
        final EnumMap enumMap = new EnumMap<>(clazz);
        final Map<String, Configurer<? extends Classifier>> map = new HashMap<>();
        for(Object obj : enumMap.entrySet()) {
            Enum entry = (Enum) obj;
            map.put(entry.name(), (Configurer<? extends Classifier>) entry);
        }
        return map;
    }
    
    private void benchmark() {
        // delegate to the benchmarking system from main experiments code. This maintains consistency across benchmarks, but it substantially quicker and therefore less reliable of a benchmark. todo talks to james about merging these
        Experiments.ExperimentalArguments args = new Experiments.ExperimentalArguments();
        args.performTimingBenchmark = true;
        benchmarkScore = Experiments.findBenchmarkTime(args);
//        long sum = 0;
//        int repeats = 30;
//        long startTime = System.nanoTime();
//        for(int i = 0; i < repeats; i++) {
//            Random random = new Random(i);
//            final double[] array = new double[1000000];
//            for(int j = 0; j < array.length; j++) {
//                array[j] = random.nextDouble();
//            }
//            Arrays.sort(array);
//        }
//        benchmarkScore = (System.nanoTime() - startTime) / repeats;
    }

    private void setup(String[] args) {
        JCommander.newBuilder()
                .addObject(this)
                .build()
                .parse(args);
        // check the state of this experiment is set up
        if(seed == null) throw new IllegalStateException("seed not set");
        if(classifierName == null) throw new IllegalStateException("classifier name not set");
        if(problemName == null) throw new IllegalStateException("problem name not set");
        if(dataDirPath == null) throw new IllegalStateException("data dir path not set");
        if(resultsDirPath == null) throw new IllegalStateException("results dir path not set");
        // default to no train time contracts
        trainTimeLimits = new ArrayList<>();
        for(String trainTimeLimitStr : trainTimeLimitStrs) {
            trainTimeLimits.add(new TimeSpan(trainTimeLimitStr));
        }
        if(trainTimeLimits.isEmpty()) {
            // add a null limit to indicate there is no limit
            trainTimeLimits.add(null);
        } else {
            // sort the train contracts in asc order
            trainTimeLimits = trainTimeLimits.stream().distinct().sorted().collect(Collectors.toList());
        }
        // default to all cpus
        if(numThreads < 1) {
            numThreads = Runtime.getRuntime().availableProcessors();
        }
    }

    public static void main(String... args) throws Exception {
        final Experiment experiment = new Experiment();
        experiment.setup(args);
        experiment.run();
    }

    private Classifier newClassifier() throws InstantiationException, IllegalAccessException {
        log.info("creating new instance of " + getClassifierName());
        return classifierLookup.get(getClassifierName()).build();
    }
    
    private final Map<String, Builder<? extends Classifier>> classifierLookup = new HashMap<>();
    private DatasetSplit split;
    private Classifier classifier;
    private final StopWatch timer = new StopWatch();
    private final StopWatch experimentTimer = new StopWatch();
    private final MemoryWatcher memoryWatcher = new MemoryWatcher();
    private final MemoryWatcher experimentMemoryWatcher = new MemoryWatcher();
    private TimeSpan trainTimeLimit;
    private long benchmarkScore;
    
    private String getExperimentResultsDirPath() {
        return StrUtils.joinPath( getResultsDirPath(), getClassifierNameInResults(), "Predictions", getProblemName());
    }
    
    private String getLockFilePath() {
        return StrUtils.joinPath(getExperimentResultsDirPath(), "fold" + getSeed() + ".lock");
    }
    
    private String getCheckpointDirPath() {
        return StrUtils.joinPath( getResultsDirPath(), getClassifierNameInResults(), "Predictions", getProblemName(), "fold" + getSeed());
    }
    
    /**
     * Runs the experiment. Make sure all fields have been set prior to this call otherwise Exceptions will be thrown accordingly.
     */
    public void run() throws Exception {
        // build the classifier
        classifier = newClassifier();
        // configure the classifier with experiment settings as necessary
        configureClassifier();
        // load the data
        log.info("loading dataset " + getProblemName());
        split = new DatasetSplit(DatasetLoading.sampleDataset(dataDirPath, problemName, seed));
        log.info("benchmarking hardware");
        benchmark();
        // make a lock to deny multiple processes from writing to the same results
        FileUtils.FileLock lock = null;
        // run the experiment
        // for each train time contract
        Assert.assertFalse(trainTimeLimits.isEmpty());
        // reset the memory watchers
        experimentTimer.resetAndStart();
        experimentMemoryWatcher.resetAndStart();
        for(TimeSpan trainTimeLimit : trainTimeLimits) {
            this.trainTimeLimit = trainTimeLimit;
            if(lock != null) {
                // unlock the previous lock
                lock.unlock();
            }
            // lock the output file to ensure only this experiment is writing results
            lock = new FileUtils.FileLock(getLockFilePath());
            // setup the contract
            setupTrainTimeContract();
            // setup checkpointing config
            setupCheckpointing();
            // train the classifier
            train();
            // test the classifier
            test();
            // stop the classifier from rebuilding on next buildClassifier call
            if(classifier instanceof Rebuildable) {
                ((Rebuildable) classifier).setRebuild(false);
            } else if(trainTimeLimits.size() > 1) {
                log.warning("cannot disable rebuild on " + getClassifierNameInResults() + ", therefore it will be rebuilt entirely for every train time contract");
            }
        }
        experimentTimer.stop();
        experimentMemoryWatcher.stop();
        log.info("experiment time: " + experimentTimer.elapsedTime());
        log.info("experiment mem: " + experimentMemoryWatcher.getMaxMemoryUsage());
        // unlock the lock file
        if(lock != null) lock.unlock();
    }

    /**
     * I.e. if we're currently preparing to run a 3h contract and previously a 1h and 2h have been run, we should check the 1h and 2h workspace for checkpoint files. If there are no checkpoint files for 2h but there are for 1h, copy them into the checkpoint dir folder to resume from the 1h contract end point.
     * @throws FileUtils.FileLock.LockException
     * @throws IOException
     */
    private void copyOverMostRecentCheckpoint() throws FileUtils.FileLock.LockException, IOException {
        // check the state of all train time contracts so far to copy over old checkpoints
        if(checkpoint) {
            // check whether the checkpoint dir is empty. If not, then we already have a checkpoint to work from, i.e. no need to copy a checkpoint from a lesser contract.
            if(!FileUtils.isEmptyDir(getCheckpointDirPath())) {
                log.info("checkpoint found in workspace");
                return;
            }
            // the target train time limit we'll be running next
            TimeSpan target = trainTimeLimit;
            // the nearest traim time limit WITH a checkpoint
            TimeSpan mostRecentTrainTimeContract = null;
            // copy this experiment to reconfigure for other contracts
            final Experiment experiment = shallowCopy();
            // for every train time limit which is less than the target
            final List<TimeSpan>
                    timeSpans = trainTimeLimits.stream().filter(timeSpan -> target.compareTo(timeSpan) <= 0).sorted(Comparator.reverseOrder()).collect(Collectors.toList());
            // examine the times in descending order, attempting to locate the most recent checkpoint
            for(TimeSpan timeSpan : timeSpans) {
                // set the dummy experiment's ttl
                experiment.trainTimeLimit = trainTimeLimit;
                // lock the checkpoints to ensure we're the only user
                try(FileUtils.FileLock lock = new FileUtils.FileLock(experiment.getLockFilePath())) {
                    // if the checkpoint dir is empty then there's no usable checkpoints
                    if(!FileUtils.isEmptyDir(experiment.getCheckpointDirPath())) {
                        // otherwise this is the most recent usable checkpoint, no need to keep searching
                        mostRecentTrainTimeContract = trainTimeLimit;
                        break;
                    }
                } catch(Exception e) {
                    // failed to lock the checkpoint dir, in use by another process.Continue looking for other checkpoints
                }
            }
            // if a previous checkpoint has been located, copy the contents into the checkpoint dir for this contract time run
            if(mostRecentTrainTimeContract != null) {
                log.info("checkpoint found in " + experiment.getClassifierNameInResults() + " workspace");
                final String src = experiment.getCheckpointDirPath();
                final String dest = getCheckpointDirPath();
                log.info("coping checkpoint contents from " + src + " to " + dest);
                Files.copy(new File(src).toPath(), new File(dest).toPath());
                // optionally remove the checkpoint now it's copied to a new location
                experiment.optionalRemoveCheckpoint();
            } else {
                log.info("no checkpoints found");
            }
        }
    }
    
    private void setResultInfo(ClassifierResults results) throws IOException {
        results.setFoldID(seed);
        String paras = results.getParas();
        if(!paras.isEmpty()) {
            paras += ",";
        }
        paras += "hostName," + hostName();
        results.setParas(paras);
        results.setBenchmarkTime(benchmarkScore);
    }
    
    private void setTrainTime(ClassifierResults results, StopWatch timer) {
        // if the build time has not been set
        if(results.getBuildTime() < 0) {
            // then set it to the build time witnessed during this experiment
            results.setBuildTime(timer.elapsedTime());
            results.setBuildPlusEstimateTime(timer.elapsedTime());
            results.setTimeUnit(TimeUnit.NANOSECONDS);
        }
    }

    private void setMemory(ClassifierResults results, MemoryWatcher memoryWatcher) {
        // if the memory usage has not been set
        if(results.getMemory() < 0) {
            // then set it to the max mem witnessed during this experiment
            results.setMemory(memoryWatcher.getMaxMemoryUsage());
        }
    }

    private void writeResults(String label, ClassifierResults results, String path) throws Exception {
        // write the train results to file, overwriting if necessary
        results.setSplit(label);
        final boolean trainResultsFileExists = new File(path).exists();
        if(trainResultsFileExists && !overwriteResults) {
            log.info(label + " results already exist");
            throw new IllegalStateException(label + " results exist");
        } else {
            log.info((trainResultsFileExists ? "overwriting" : "writing") + " " + label + " results");
            results.writeFullResultsToFile(path);
        }
    }

    private void setupTrainTimeContract() {
        if(trainTimeLimit != null) {
            // there is a contract
            if(classifier instanceof TrainTimeContractable) {
                log.info("setting classifier train contract to " + trainTimeLimit);
                ((TrainTimeContractable) classifier).setTrainTimeLimit(trainTimeLimit.inNanos());
            } else {
                throw new IllegalStateException("classifier cannot handle train time contract");
            }
        }  // else there is no contract, proceed as is
    }

    private void configureClassifier() {
        log.info("configuring " + getClassifierName());
        // set estimate train error
        if(evaluateClassifier) {
            if(classifier instanceof TrainEstimateable) {
                log.info("setting classifier to estimate train error");
                ((TrainEstimateable) classifier).setEstimateOwnPerformance(true);
            } else {
                throw new IllegalStateException("classifier cannot evaluate the train error");
            }
        }
        // set checkpointing
        if(checkpoint) {
            if(classifier instanceof Checkpointable) {
                log.info("setting classifier to checkpoint to " + getCheckpointDirPath());
                ((Checkpointable) classifier).setCheckpointPath(getCheckpointDirPath());
            } else {
                throw new IllegalStateException("classifier cannot checkpoint");
            }
        }
        // set log level
        if(classifier instanceof Loggable) {
            log.info("setting classifier log level to " + getLogLevel());
            ((Loggable) classifier).setLogLevel(getLogLevel());
        } else if(classifier instanceof EnhancedAbstractClassifier) {
            boolean debug = !getLogLevel().equals(OFF);
            log.info("setting classifier debug to " + debug);
            ((EnhancedAbstractClassifier) classifier).setDebug(debug);
        } else {
            if(!getLogLevel().equals(Level.OFF)) {
                log.info("classifier does not support logging");
            }
        }
        // set seed
        if(classifier instanceof Randomizable) {
            log.info("setting classifier seed to " + getSeed());
            ((Randomizable) classifier).setSeed(getSeed());
        } else {
            log.info("classifier does not accept a seed");
        }
        // set threads
        if(classifier instanceof MultiThreadable) {
            log.info("setting classifier to use " + getNumThreads() + " threads");
            ((MultiThreadable) classifier).enableMultiThreading(getNumThreads());
        } else if(getNumThreads() != 1) {
            log.info("classifier cannot use multiple threads");
        }
    }


    public String getResultsDirPath() {
        return resultsDirPath;
    }

    public String getClassifierNameInResults() {
        String classifierNameInResults = classifierName;
        if(trainTimeLimit != null) {
            classifierNameInResults += "_" + trainTimeLimit;
        }
        return classifierNameInResults;
    }

    public String getProblemName() {
        return problemName;
    }

    public Integer getSeed() {
        return seed;
    }

    public String getDataDirPath() {
        return dataDirPath;
    }
    
    public String getTestFilePath() {
        return StrUtils.joinPath(getExperimentResultsDirPath(), "testFold" + seed + ".csv");
    }

    public String getTrainFilePath() {
        return StrUtils.joinPath(getExperimentResultsDirPath(), "trainFold" + seed + ".csv");
    }
    
    private void train() throws Exception {
        // build the classifier
        log.info("training classifier");
        timer.start();
        memoryWatcher.start();
        // prompt garbage collection to provide a clean slate before building
        System.gc();
        if(classifier instanceof TSClassifier) {
            ((TSClassifier) classifier).buildClassifier(split.getTrainDataTS());
        } else {
            classifier.buildClassifier(split.getTrainDataArff());
        }
        // prompt garbage collection to obtain at least one memory usage reading during training
        System.gc();
        timer.stop();
        memoryWatcher.stop();
        log.info("train time: " + timer.elapsedTime());
        log.info("train mem: " + memoryWatcher.getMaxMemoryUsage());
        // if estimating the train error then write out train results
        if(evaluateClassifier) {
            final ClassifierResults trainResults = ((TrainEstimateable) classifier).getTrainResults();
            ResultUtils.setInfo(trainResults, classifier, split.getTrainDataArff()); // todo change to ts
            setResultInfo(trainResults);
            setTrainTime(trainResults, timer);
            setMemory(trainResults, memoryWatcher);
            trainResults.findAllStatsOnce();
            log.info("train results: ");
            log.info(trainResults.writeSummaryResultsToString());
            writeResults("train", trainResults, getTrainFilePath());
        }
    }
    
    private void test() throws Exception {
        // if only training then skip the test phase
        if(trainOnly) {
            log.info("skipping testing classifier");
            return;
        }
        // test the classifier
        log.info("testing classifier");
        timer.resetAndStart();
        memoryWatcher.resetAndStart();
        final ClassifierResults testResults = new ClassifierResults();
        // todo change to ts
        //            if(classifier instanceof TSClassifier) {
        //                ClassifierTools.addPredictions((TSClassifier) classifier, split.getTestDataTS(), testResults, new Random(seed));
        //            } else {
        ClassifierTools.addPredictions(classifier, split.getTestDataArff(), testResults, new Random(seed));
        //            }
        timer.stop();
        memoryWatcher.stop();
        log.info("test time: " + timer.elapsedTime());
        log.info("test mem: " + memoryWatcher.getMaxMemoryUsage());
        ResultUtils.setInfo(testResults, classifier, split.getTrainDataArff()); // todo ts version
        setResultInfo(testResults);
        setTrainTime(testResults, timer);
        setMemory(testResults, memoryWatcher);
        log.info("test results: ");
        log.info(testResults.writeSummaryResultsToString());
        writeResults("test", testResults, getTestFilePath());
        if(trainTimeLimit == null) {
            log.info("experiment complete");
        } else {
            log.info("train time contract " + trainTimeLimit + " experiment complete");
        }
    }

    public String getClassifierName() {
        return classifierName;
    }

    public Level getLogLevel() {
        return logLevel;
    }

    public int getNumThreads() {
        return numThreads;
    }
    
    private void setupCheckpointing() throws FileUtils.FileLock.LockException, IOException {
        if(checkpoint) {
            if(classifier instanceof Checkpointable) {
                // the copy over the most suitable checkpoint from another run if exists
                copyOverMostRecentCheckpoint();
                log.info("setting checkpoint path for " + getClassifierName() + " to " + getCheckpointDirPath());
                ((Checkpointable) classifier).setCheckpointPath(getCheckpointDirPath());
            } else {
                log.info(getClassifierName() + " cannot produce checkpoints");
            }
        }
    }
    
    private void optionalRemoveCheckpoint() {
        // if removing checkpoints then we can remove the most recent checkpoint as it has been copied over to the next contract
        if(removeCheckpoint) {
            log.info("removing previous checkpoint: " + getCheckpointDirPath());
            try {
                Files.delete(Paths.get(getCheckpointDirPath()));
            } catch(IOException e) {
                System.err.println("failed to remove checkpoint at " + getCheckpointDirPath());
                System.err.println(e);
            }
        }
    }
}
