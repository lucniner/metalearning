package at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors;

import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import at.hwl.machinelearning.ass3.metalearning.utils.FeaturePair;

public class DataSetNameExtractor extends AbstractFeatureExtractor {


  public DataSetNameExtractor(DataSetInstance instance) {
    super(instance);
  }

  @Override
  public FeaturePair call() throws Exception {
    final String relationName = instance.getWekaInstance().relationName();
    return new FeaturePair("instance_name", relationName);
  }
}
