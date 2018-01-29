package at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors.general;

import at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors.AbstractFeatureExtractor;
import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import at.hwl.machinelearning.ass3.metalearning.utils.FeaturePair;
import at.hwl.machinelearning.ass3.metalearning.utils.SharedConstants;
import weka.core.Instances;


public class NumClassFeatureExtractor extends AbstractFeatureExtractor {


  public NumClassFeatureExtractor(DataSetInstance instance) {
    super(instance);
  }

  @Override
  public FeaturePair call() {
    final double numberOfClasses = extractNumberOfClasses();
    return new FeaturePair(SharedConstants.NUMBER_OF_CLASSES, String.valueOf(numberOfClasses));
  }


  private double extractNumberOfClasses() {
    final Instances wekaInstance = instance.getWekaInstance();
    final int classIndex = wekaInstance.classIndex();
    return wekaInstance.numDistinctValues(classIndex);
  }
}
