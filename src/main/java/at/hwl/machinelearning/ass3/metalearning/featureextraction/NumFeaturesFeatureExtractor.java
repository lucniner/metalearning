package at.hwl.machinelearning.ass3.metalearning.featureextraction;

import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import at.hwl.machinelearning.ass3.metalearning.utils.FeaturePair;
import at.hwl.machinelearning.ass3.metalearning.utils.SharedConstants;
import weka.core.Instances;

class NumFeaturesFeatureExtractor extends AbstractFeatureExtractor {


  protected NumFeaturesFeatureExtractor(DataSetInstance instance) {
    super(instance);
  }

  @Override
  public FeaturePair call() {
    final double numberOfFeatures = extractNumberOfFeatures();
    return new FeaturePair(SharedConstants.NUMBER_OF_FEATURES, String.valueOf(numberOfFeatures));
  }


  private double extractNumberOfFeatures() {
    final Instances wekaInstance = instance.getWekaInstance();
    return wekaInstance.numAttributes();
  }
}
