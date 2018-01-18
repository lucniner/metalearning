package at.hwl.machinelearning.ass3.metalearning.classification.instanceclassification;

import at.hwl.machinelearning.ass3.metalearning.utils.ClassificationResult;
import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import weka.classifiers.Classifier;
import weka.classifiers.rules.M5Rules;

/**
 * Class for building and using a 0-R classifier. Predicts the mean (for a numeric class) or the mode (for a nominal class).
 */
class ZeroRClassifier extends AbstractClassifier {
  ZeroRClassifier(DataSetInstance instance, int trainTestSplitPercent) {
    super(instance, trainTestSplitPercent);
  }

  @Override
  Classifier getClassifier() {
    return new M5Rules();
  }

  @Override
  public ClassificationResult call() throws Exception {
    final double accuracy = classify();
    return new ClassificationResult("zeror", accuracy);
  }
}
