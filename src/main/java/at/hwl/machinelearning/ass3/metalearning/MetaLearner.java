package at.hwl.machinelearning.ass3.metalearning;

import at.hwl.machinelearning.ass3.metalearning.classification.MetaClassifierRunner;
import at.hwl.machinelearning.ass3.metalearning.classification.TrainTestSplitClassificationRunner;
import at.hwl.machinelearning.ass3.metalearning.datahandling.InstanceCreator;
import at.hwl.machinelearning.ass3.metalearning.datahandling.MetaResultWriter;
import at.hwl.machinelearning.ass3.metalearning.featureextraction.FeatureExtractorRunner;
import at.hwl.machinelearning.ass3.metalearning.utils.ClassificationAccuracyResult;
import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstances;
import at.hwl.machinelearning.ass3.metalearning.utils.FeaturePair;
import at.hwl.machinelearning.ass3.metalearning.utils.FeaturePairs;
import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.stream.Collectors;

public class MetaLearner {

  private final Map<String, String> bestClassifierPerDataSet = new HashMap<>();
  private final Map<String, List<String>> featureValuesPerDataSet = new HashMap<>();
  private final List<String> featureNames = new LinkedList<>();

  private final ExecutorService executorService;
  private final InstanceCreator instanceCreator;
  private final InstanceCreator metaResultCreator;
  private final String metaExtractionResultFile;

  MetaLearner(
      ExecutorService executorService,
      InstanceCreator instanceCreator,
      InstanceCreator metaResultCreator,
      String metaExtractionResultFile) {
    this.executorService = executorService;
    this.instanceCreator = instanceCreator;
    this.metaResultCreator = metaResultCreator;
    this.metaExtractionResultFile = metaExtractionResultFile;
  }

  public void learn() throws Exception {
    final DataSetInstances instances = instanceCreator.loadInstances();
    final List<DataSetInstance> datasets = instances.getAllInstances();
    calculate(datasets);
    writeNewDataSet();
    calculateMetaLearningAlgorithm();
  }

  private void calculate(final List<DataSetInstance> instances)
      throws ExecutionException, InterruptedException {
    for (final DataSetInstance instance : instances) {
      final Future<FeaturePairs> pairs = calculateFeatures(instance);
      final Future<ClassificationAccuracyResult> result = calculateAccuracy(instance);
      final FeaturePairs featurePairs = pairs.get();
      writeAccuracyResult(instance, result.get());
      writeFeatureResult(instance, featurePairs);
      writeFeatureNamesIfNeeded(featurePairs);
    }
  }

  private Future<FeaturePairs> calculateFeatures(final DataSetInstance instance) {
    final FeatureExtractorRunner extractor = new FeatureExtractorRunner(executorService, instance);
    return executorService.submit(extractor);
  }

  private Future<ClassificationAccuracyResult> calculateAccuracy(final DataSetInstance instance) {
    final TrainTestSplitClassificationRunner runner =
        new TrainTestSplitClassificationRunner(executorService, instance);
    return executorService.submit(runner);
  }

  private void writeNewDataSet() throws IOException {
    final MetaResultWriter writer =
        new MetaResultWriter(bestClassifierPerDataSet, featureValuesPerDataSet, featureNames);
    writer.writeResultsToFile(metaExtractionResultFile);
  }

  private void calculateMetaLearningAlgorithm() throws Exception {
    final DataSetInstance instance = metaResultCreator.getSingleInstance();
    final MetaClassifierRunner runner = new MetaClassifierRunner(executorService, instance);
    final Future<ClassificationAccuracyResult> future = executorService.submit(runner);
    final ClassificationAccuracyResult result = future.get();
    result.getAllClassificationResults().entrySet().forEach(System.out::println);
  }

  private void writeFeatureNamesIfNeeded(final FeaturePairs pairs) {
    if (featureNames.isEmpty()) {
      final List<String> featureKeys =
          pairs
              .getFeaturePairs()
              .stream()
              .map(FeaturePair::getFeatureName)
              .collect(Collectors.toList());
      featureNames.addAll(featureKeys);
    }
  }

  private void writeAccuracyResult(
      final DataSetInstance instance, ClassificationAccuracyResult result) {
    bestClassifierPerDataSet.put(instance.getDataSetLocation(), result.getBestClassifier());
  }

  private void writeFeatureResult(final DataSetInstance instance, FeaturePairs pairs) {
    final List<String> featureValues =
        pairs
            .getFeaturePairs()
            .stream()
            .map(FeaturePair::getFeatureValue)
            .collect(Collectors.toList());
    featureValuesPerDataSet.put(instance.getDataSetLocation(), featureValues);
  }
}
