package at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors.modelbased;

import at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors.AbstractFeatureExtractor;
import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import at.hwl.machinelearning.ass3.metalearning.utils.FeaturePair;
import at.hwl.machinelearning.ass3.metalearning.utils.SharedConstants;
import weka.classifiers.trees.REPTree;
import weka.core.Instances;

/**
 * <h4>About this class</h4>
 *
 * <p>Description of this class</p>
 *
 * @author David Molnar
 * @version 0.1.0
 * @since 0.1.0
 */
public class REPTreeSizeFeatureExtractor extends AbstractFeatureExtractor {

  public REPTreeSizeFeatureExtractor(
      DataSetInstance instance) {
    super(instance);
  }

  @Override
  public FeaturePair call() throws Exception {
    final double numberOfClasses = extractMaxDepth();
    return new FeaturePair(SharedConstants.REP_TREE_SIZE, String.valueOf(numberOfClasses));
  }

  private double extractMaxDepth() throws Exception {
    final Instances wekaInstance = instance.getWekaInstance();
    final REPTree cls = new REPTree();

    cls.buildClassifier(wekaInstance);

    return cls.numNodes();
  }
}
