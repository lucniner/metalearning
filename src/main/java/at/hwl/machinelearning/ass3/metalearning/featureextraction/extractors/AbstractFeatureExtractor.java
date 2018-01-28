package at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors;

import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import at.hwl.machinelearning.ass3.metalearning.utils.FeaturePair;
import java.util.concurrent.Callable;

public abstract class AbstractFeatureExtractor implements Callable<FeaturePair> {

  protected final DataSetInstance instance;


  protected AbstractFeatureExtractor(DataSetInstance instance) {
    this.instance = instance;
  }
}
