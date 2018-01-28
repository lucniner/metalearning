package at.hwl.machinelearning.ass3.metalearning.classification.classifiers;

import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

/**
 * Class for constructing a forest of random trees.
 */
public class RandomForestClassifier implements IClassifiable {

  @Override
  public Classifier getClassifier() {
    return new RandomForest();
  }

  @Override
  public Instances prepareInstance(Instances instances) {
    return instances;
  }

  @Override
  public String getName() {
    return "random_forrest";
  }

}
