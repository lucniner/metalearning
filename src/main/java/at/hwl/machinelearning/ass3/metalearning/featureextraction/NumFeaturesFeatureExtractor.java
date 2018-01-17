package at.hwl.machinelearning.ass3.metalearning.featureextraction;

import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import at.hwl.machinelearning.ass3.metalearning.utils.FeaturePair;
import at.hwl.machinelearning.ass3.metalearning.utils.SharedConstants;

class NumFeaturesFeatureExtractor extends AbstractFeatureExtractor {


  protected NumFeaturesFeatureExtractor(DataSetInstance instance) {
    super(instance);
  }

  @Override
  public FeaturePair call() throws Exception {
    return new FeaturePair(SharedConstants.NUMBER_OF_FEATURES, 0.0);
  }
}
