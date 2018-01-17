package at.hwl.machinelearning.ass3.metalearning.featureextraction;

import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import at.hwl.machinelearning.ass3.metalearning.utils.FeaturePair;
import at.hwl.machinelearning.ass3.metalearning.utils.SharedConstants;
import weka.core.Instances;


class NumClassFeatureExtractor extends AbstractFeatureExtractor {


  NumClassFeatureExtractor(DataSetInstance instance) {
    super(instance);
  }

  @Override
  public FeaturePair call() {
    final double numberOfClasses = extractNumberOfClasses();
    return new FeaturePair(SharedConstants.NUMBER_OF_CLASSES, numberOfClasses);
  }


  private double extractNumberOfClasses() {
    final Instances wekaInstance = instance.getWekaInstance();
    final int classIndex = wekaInstance.classIndex();
    return wekaInstance.numDistinctValues(classIndex);
  }
}
