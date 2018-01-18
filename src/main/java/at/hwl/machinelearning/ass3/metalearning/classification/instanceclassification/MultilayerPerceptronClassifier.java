package at.hwl.machinelearning.ass3.metalearning.classification.instanceclassification;

import at.hwl.machinelearning.ass3.metalearning.utils.ClassificationResult;
import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;

/**
 * A Classifier that uses backpropagation to classify instances.
 */
class MultilayerPerceptronClassifier extends AbstractClassifier {
  MultilayerPerceptronClassifier(DataSetInstance instance, int trainTestSplitPercent) {
    super(instance, trainTestSplitPercent);
  }

  @Override
  Classifier getClassifier() {
    return new MultilayerPerceptron();
  }

  @Override
  public ClassificationResult call() throws Exception {
    final double accuracy = classify();
    return new ClassificationResult("multilayer_perceptron", accuracy);
  }
}
