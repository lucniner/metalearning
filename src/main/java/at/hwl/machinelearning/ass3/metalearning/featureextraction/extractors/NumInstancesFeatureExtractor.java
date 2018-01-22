package at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors;

import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import at.hwl.machinelearning.ass3.metalearning.utils.FeaturePair;
import at.hwl.machinelearning.ass3.metalearning.utils.SharedConstants;
import weka.core.Instances;


public class NumInstancesFeatureExtractor extends AbstractFeatureExtractor {

  public NumInstancesFeatureExtractor(DataSetInstance instance) {
    super(instance);
  }

  @Override
  public FeaturePair call() {
    final double numberOfInstances = extractNumberOfInstances();
    return new FeaturePair(SharedConstants.NUMBER_OF_INSTANCES, String.valueOf(numberOfInstances));
  }

  private double extractNumberOfInstances() {
    final Instances wekaInstance = instance.getWekaInstance();
    return wekaInstance.numInstances();
  }
}
