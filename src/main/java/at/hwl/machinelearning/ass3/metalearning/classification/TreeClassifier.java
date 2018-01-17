package at.hwl.machinelearning.ass3.metalearning.classification;

import at.hwl.machinelearning.ass3.metalearning.utils.ClassificationResult;
import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;

public class TreeClassifier extends AbstractClassifier {

  protected TreeClassifier(DataSetInstance instance) {
    super(instance);
  }

  @Override
  public ClassificationResult call() throws Exception {
    return new ClassificationResult("tree", 0.97);
  }
}
