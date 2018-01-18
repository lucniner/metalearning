package at.hwl.machinelearning.ass3.metalearning.classification.instanceclassification;

import at.hwl.machinelearning.ass3.metalearning.utils.ClassificationResult;
import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.KStar;

/**
 * K* is an instance-based classifier, that is the class of a test instance is based upon the class of those training instances similar to it, as determined by some similarity function. It differs from other instance-based learners in that it uses an entropy-based distance function.
 */
class KStarClassifier extends AbstractClassifier {

  KStarClassifier(DataSetInstance instance, int trainTestSplitPercent) {
    super(instance, trainTestSplitPercent);
  }

  @Override
  Classifier getClassifier() {
    return new KStar();
  }

  @Override
  public ClassificationResult call() throws Exception {
    final double accuracy = classify();
    return new ClassificationResult("k_star", accuracy);
  }


}
