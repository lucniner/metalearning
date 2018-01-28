package at.hwl.machinelearning.ass3.metalearning.classification.classifiers;

import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;

/**
 * A Classifier that uses backpropagation to classify instances.
 */
public class MultilayerPerceptronClassifier implements IClassifiable {

  @Override
  public Classifier getClassifier() {
    return new MultilayerPerceptron();
  }

  @Override
  public Instances prepareInstance(Instances instances) {
    return instances;
  }

  @Override
  public String getName() {
    return "multi_layer_perceptron";
  }

}
