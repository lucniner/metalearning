package at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors;

import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import at.hwl.machinelearning.ass3.metalearning.utils.FeaturePair;
import at.hwl.machinelearning.ass3.metalearning.utils.SharedConstants;
import java.util.Enumeration;
import weka.core.Attribute;
import weka.core.Instances;

public class NumMissingValuesFeatureExtractor extends AbstractFeatureExtractor {

  public NumMissingValuesFeatureExtractor(
      DataSetInstance instance) {
    super(instance);
  }

  @Override
  public FeaturePair call() {
    final int numberOfMissingValues = extractNumberOfMissingValues();
    return new FeaturePair(SharedConstants.NUMBER_OF_MISSING_VALUES, String.valueOf(numberOfMissingValues));
  }

  private int extractNumberOfMissingValues() {
    final Instances wekaInstance = instance.getWekaInstance();
    final Enumeration<Attribute> attributes = wekaInstance.enumerateAttributes();

    int missingValuesCount = 0;
    while (attributes.hasMoreElements()) {
      missingValuesCount += wekaInstance
          .attributeStats(attributes.nextElement().index()).missingCount;
    }

    return missingValuesCount;
  }
}
