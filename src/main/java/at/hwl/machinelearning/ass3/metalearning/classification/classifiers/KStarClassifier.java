package at.hwl.machinelearning.ass3.metalearning.classification.classifiers;

import weka.classifiers.Classifier;
import weka.classifiers.lazy.KStar;
import weka.core.Instances;

/**
 * K* is an instance-based classifier, that is the class of a test instance is based upon the class of those training instances similar to it, as determined by some similarity function. It differs from other instance-based learners in that it uses an entropy-based distance function.
 */
public class KStarClassifier implements IClassifiable {


  @Override
  public Classifier getClassifier() {
    return new KStar();
  }

  @Override
  public Instances prepareInstance(Instances instances) {
    return instances;
  }

  @Override
  public String getName() {
    return "kstar";
  }


}
