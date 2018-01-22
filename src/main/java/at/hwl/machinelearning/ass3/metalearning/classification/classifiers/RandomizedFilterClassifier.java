package at.hwl.machinelearning.ass3.metalearning.classification.classifiers;

import weka.classifiers.Classifier;
import weka.classifiers.meta.RandomizableFilteredClassifier;
import weka.core.Instances;

public class RandomizedFilterClassifier implements IClassifyable {
  @Override
  public Classifier getClassifier() {
    return new RandomizableFilteredClassifier();
  }

  @Override
  public Instances prepareInstance(Instances instances) {
    return instances;
  }

  @Override
  public String getName() {
    return "randomizable_filter";
  }
}
