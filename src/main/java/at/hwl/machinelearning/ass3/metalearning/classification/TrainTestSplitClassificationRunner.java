package at.hwl.machinelearning.ass3.metalearning.classification;

import at.hwl.machinelearning.ass3.metalearning.classification.classifiers.BayesNetClassifier;
import at.hwl.machinelearning.ass3.metalearning.classification.classifiers.KStarClassifier;
import at.hwl.machinelearning.ass3.metalearning.classification.classifiers.MultilayerPerceptronClassifier;
import at.hwl.machinelearning.ass3.metalearning.classification.classifiers.REPTreeClassifier;
import at.hwl.machinelearning.ass3.metalearning.classification.classifiers.RandomForestClassifier;
import at.hwl.machinelearning.ass3.metalearning.classification.classifiers.RandomizedFilterClassifier;
import at.hwl.machinelearning.ass3.metalearning.classification.classifiers.ZeroRClassifier;
import at.hwl.machinelearning.ass3.metalearning.utils.ClassificationAccuracyResult;
import at.hwl.machinelearning.ass3.metalearning.utils.ClassificationResult;
import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

public class TrainTestSplitClassificationRunner implements Callable<ClassificationAccuracyResult> {

  private static final int TRAIN_TEST_SPLIT_PERCENT = 80;
  private final ClassificationAccuracyResult results = new ClassificationAccuracyResult();

  private final Collection<Callable<ClassificationResult>> classifiers = new ArrayList<>();

  private final ExecutorService executorService;
  private final DataSetInstance dataSetInstance;

  public TrainTestSplitClassificationRunner(
      ExecutorService executorService, DataSetInstance dataSetInstance) {
    this.executorService = executorService;
    this.dataSetInstance = dataSetInstance;

    this.initializeClassifiers();
  }

  private void initializeClassifiers() {
    classifiers.add(
        new TrainTestSplitClassifier(
            dataSetInstance, TRAIN_TEST_SPLIT_PERCENT, new KStarClassifier()));
    classifiers.add(
        new TrainTestSplitClassifier(
            dataSetInstance, TRAIN_TEST_SPLIT_PERCENT, new MultilayerPerceptronClassifier()));
    classifiers.add(
        new TrainTestSplitClassifier(
            dataSetInstance, TRAIN_TEST_SPLIT_PERCENT, new RandomForestClassifier()));
    classifiers.add(
        new TrainTestSplitClassifier(
            dataSetInstance, TRAIN_TEST_SPLIT_PERCENT, new REPTreeClassifier()));
    classifiers.add(
        new TrainTestSplitClassifier(
            dataSetInstance, TRAIN_TEST_SPLIT_PERCENT, new ZeroRClassifier()));
    classifiers.add(
        new TrainTestSplitClassifier(
            dataSetInstance, TRAIN_TEST_SPLIT_PERCENT, new RandomizedFilterClassifier()));
    classifiers.add(
        new TrainTestSplitClassifier(
            dataSetInstance, TRAIN_TEST_SPLIT_PERCENT, new BayesNetClassifier()));
  }

  @Override
  public ClassificationAccuracyResult call() throws Exception {
    return calculateClassificationResult();
  }

  private ClassificationAccuracyResult calculateClassificationResult()
      throws ExecutionException, InterruptedException {
    final List<Future<ClassificationResult>> futures = executorService.invokeAll(classifiers);
    for (Future<ClassificationResult> c : futures) {
      results.addResult(c.get());
    }
    return results;
  }
}
