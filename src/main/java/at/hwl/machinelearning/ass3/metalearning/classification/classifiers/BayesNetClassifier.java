package at.hwl.machinelearning.ass3.metalearning.classification.classifiers;

import at.hwl.machinelearning.ass3.metalearning.classification.preprocessing.NumericToNominalPreprocessor;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesNet;
import weka.core.Instances;

public class BayesNetClassifier implements IClassifyable {


  @Override
  public Classifier getClassifier() {
    return new BayesNet();
  }

  @Override
  public Instances prepareInstance(final Instances instances) {
    final NumericToNominalPreprocessor preprocessor = new NumericToNominalPreprocessor(instances);
    return preprocessor.preprocess();
  }

  @Override
  public String getName() {
    return "bayes_net";
  }


}
