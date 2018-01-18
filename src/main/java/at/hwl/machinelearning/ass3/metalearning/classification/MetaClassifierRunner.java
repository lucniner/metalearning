package at.hwl.machinelearning.ass3.metalearning.classification;

import at.hwl.machinelearning.ass3.metalearning.classification.classifiers.KStarClassifier;
import at.hwl.machinelearning.ass3.metalearning.classification.classifiers.MultilayerPerceptronClassifier;
import at.hwl.machinelearning.ass3.metalearning.classification.classifiers.RandomForestClassifier;
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

public class MetaClassifierRunner implements Callable<ClassificationAccuracyResult> {

  private final ClassificationAccuracyResult results = new ClassificationAccuracyResult();
  private final Collection<Callable<ClassificationResult>> classifiers = new ArrayList<>();
  private final ExecutorService executorService;
  private final DataSetInstance dataSetInstance;

  public MetaClassifierRunner(ExecutorService executorService, DataSetInstance dataSetInstance) {
    this.executorService = executorService;
    this.dataSetInstance = dataSetInstance;

    this.initializeClassifiers();
  }

  private void initializeClassifiers() {
    classifiers.add(new MetaClassifier(dataSetInstance, new KStarClassifier()));
    classifiers.add(new MetaClassifier(dataSetInstance, new MultilayerPerceptronClassifier()));
    classifiers.add(new MetaClassifier(dataSetInstance, new RandomForestClassifier()));
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
