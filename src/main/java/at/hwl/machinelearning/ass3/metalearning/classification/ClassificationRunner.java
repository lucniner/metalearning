package at.hwl.machinelearning.ass3.metalearning.classification;

import at.hwl.machinelearning.ass3.metalearning.utils.ClassificationAccuracyResult;
import at.hwl.machinelearning.ass3.metalearning.utils.ClassificationResult;
import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

public class ClassificationRunner implements Callable<ClassificationAccuracyResult> {

  final ClassificationAccuracyResult results = new ClassificationAccuracyResult();


  private final ExecutorService executorService;
  private final DataSetInstance dataSetInstance;

  public ClassificationRunner(ExecutorService executorService, DataSetInstance dataSetInstance) {
    this.executorService = executorService;
    this.dataSetInstance = dataSetInstance;
  }

  @Override
  public ClassificationAccuracyResult call() throws Exception {

    final AbstractClassifier classifier = new TreeClassifier(dataSetInstance);

    final Collection<Callable<ClassificationResult>> classifiers = new ArrayList<>();
    classifiers.add(classifier);

    final List<Future<ClassificationResult>> futures = executorService.invokeAll(classifiers);
    for (Future<ClassificationResult> c : futures) {
      results.addResult(c.get());
    }


    return results;
  }
}
