package at.hwl.machinelearning.ass3.metalearning.classification.classifiers;

import weka.classifiers.Classifier;
import weka.classifiers.rules.ZeroR;
import weka.core.Instances;

/**
 * Class for building and using a 0-R classifier. Predicts the mean (for a numeric class) or the mode (for a nominal class).
 */
public class ZeroRClassifier implements IClassifiable {

  @Override
  public Classifier getClassifier() {
    return new ZeroR();
  }

  @Override
  public Instances prepareInstance(Instances instances) {
    return instances;
  }

  @Override
  public String getName() {
    return "zeroR";
  }
}
