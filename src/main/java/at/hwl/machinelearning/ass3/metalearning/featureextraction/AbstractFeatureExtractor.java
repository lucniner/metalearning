package at.hwl.machinelearning.ass3.metalearning.featureextraction;

import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import at.hwl.machinelearning.ass3.metalearning.utils.FeaturePair;

import java.util.concurrent.Callable;

abstract class AbstractFeatureExtractor implements Callable<FeaturePair> {

  final DataSetInstance instance;


  protected AbstractFeatureExtractor(DataSetInstance instance) {
    this.instance = instance;
  }
}
