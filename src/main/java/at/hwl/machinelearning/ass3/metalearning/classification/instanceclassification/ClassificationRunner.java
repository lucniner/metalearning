package at.hwl.machinelearning.ass3.metalearning.classification.instanceclassification;

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

public class ClassificationRunner implements Callable<ClassificationAccuracyResult> {

  private static final int TRAIN_TEST_SPLIT_PERCENT = 80;
  private final ClassificationAccuracyResult results = new ClassificationAccuracyResult();

  private final Collection<Callable<ClassificationResult>> classifiers = new ArrayList<>();


  private final ExecutorService executorService;
  private final DataSetInstance dataSetInstance;

  public ClassificationRunner(ExecutorService executorService, DataSetInstance dataSetInstance) {
    this.executorService = executorService;
    this.dataSetInstance = dataSetInstance;

    this.initializeClassifiers();
  }


  private void initializeClassifiers() {
    classifiers.add(new KStarClassifier(dataSetInstance, TRAIN_TEST_SPLIT_PERCENT));
    classifiers.add(new MultilayerPerceptronClassifier(dataSetInstance, TRAIN_TEST_SPLIT_PERCENT));
    classifiers.add(new RandomForestClassifier(dataSetInstance, TRAIN_TEST_SPLIT_PERCENT));
    classifiers.add(new REPTreeClassifier(dataSetInstance, TRAIN_TEST_SPLIT_PERCENT));
    classifiers.add(new ZeroRClassifier(dataSetInstance, TRAIN_TEST_SPLIT_PERCENT));
    classifiers.add(new M5RulesClassifier(dataSetInstance, TRAIN_TEST_SPLIT_PERCENT));
  }

  @Override
  public ClassificationAccuracyResult call() throws Exception {
    return calculateClassificationResult();
  }

  private ClassificationAccuracyResult calculateClassificationResult() throws ExecutionException, InterruptedException {
    final List<Future<ClassificationResult>> futures = executorService.invokeAll(classifiers);
    for (Future<ClassificationResult> c : futures) {
      results.addResult(c.get());
    }
    return results;
  }
}
