package at.hwl.machinelearning.ass3.metalearning.featureextraction;

import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import at.hwl.machinelearning.ass3.metalearning.utils.FeaturePair;

class DataSetNameExtractor extends AbstractFeatureExtractor {


  DataSetNameExtractor(DataSetInstance instance) {
    super(instance);
  }

  @Override
  public FeaturePair call() throws Exception {
    final String relationName = instance.getWekaInstance().relationName();
    return new FeaturePair("instance_name", relationName);
  }
}
