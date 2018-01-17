package at.hwl.machinelearning.ass3.metalearning.featureextraction;

import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import at.hwl.machinelearning.ass3.metalearning.utils.FeaturePair;
import at.hwl.machinelearning.ass3.metalearning.utils.SharedConstants;


class NumClassFeatureExtractor extends AbstractFeatureExtractor {


  protected NumClassFeatureExtractor(DataSetInstance instance) {
    super(instance);
  }

  @Override
  public FeaturePair call() throws Exception {
    return new FeaturePair(SharedConstants.NUMBER_OF_CLASSES, 2.0);
  }
}
