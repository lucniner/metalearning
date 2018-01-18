package at.hwl.machinelearning.ass3.metalearning.classification.instanceclassification;

import at.hwl.machinelearning.ass3.metalearning.utils.ClassificationResult;
import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;

/**
 * Class for constructing a forest of random trees.
 */
class RandomForestClassifier extends AbstractClassifier {
  RandomForestClassifier(DataSetInstance instance, int trainTestSplitPercent) {
    super(instance, trainTestSplitPercent);
  }

  @Override
  Classifier getClassifier() {
    return new RandomForest();
  }

  @Override
  public ClassificationResult call() throws Exception {
    final double accuracy = classify();
    return new ClassificationResult("random_forest", accuracy);
  }
}
