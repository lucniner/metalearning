package at.hwl.machinelearning.ass3.metalearning.classification.instanceclassification;

import at.hwl.machinelearning.ass3.metalearning.utils.ClassificationResult;
import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import weka.classifiers.Classifier;
import weka.classifiers.trees.REPTree;

/**
 * Fast decision tree learner. Builds a decision/regression tree using information gain/variance and prunes it using reduced-error pruning (with backfitting). Only sorts values for numeric attributes once. Missing values are dealt with by splitting the corresponding instances into pieces
 */
class REPTreeClassifier extends AbstractClassifier {
  REPTreeClassifier(DataSetInstance instance, int trainTestSplitPercent) {
    super(instance, trainTestSplitPercent);
  }

  @Override
  Classifier getClassifier() {
    return new REPTree();
  }

  @Override
  public ClassificationResult call() throws Exception {
    final double accuracy = classify();
    return new ClassificationResult("rep_tree", accuracy);
  }
}
