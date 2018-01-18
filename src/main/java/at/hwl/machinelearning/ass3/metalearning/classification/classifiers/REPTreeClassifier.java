package at.hwl.machinelearning.ass3.metalearning.classification.classifiers;

import weka.classifiers.Classifier;
import weka.classifiers.trees.REPTree;
import weka.core.Instances;

/**
 * Fast decision tree learner. Builds a decision/regression tree using information gain/variance and prunes it using reduced-error pruning (with backfitting). Only sorts values for numeric attributes once. Missing values are dealt with by splitting the corresponding instances into pieces
 */
public class REPTreeClassifier implements IClassifyable {

  @Override
  public Classifier getClassifier() {
    return new REPTree();
  }

  @Override
  public Instances prepareInstance(Instances instances) {
    return instances;
  }

  @Override
  public String getName() {
    return "reptree";
  }

}
