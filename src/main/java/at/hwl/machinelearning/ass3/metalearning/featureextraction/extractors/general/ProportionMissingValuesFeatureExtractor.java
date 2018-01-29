package at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors.general;

import at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors.AbstractFeatureExtractor;
import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import at.hwl.machinelearning.ass3.metalearning.utils.FeaturePair;
import at.hwl.machinelearning.ass3.metalearning.utils.SharedConstants;
import java.util.Enumeration;
import weka.core.Attribute;
import weka.core.Instances;

public class ProportionMissingValuesFeatureExtractor extends AbstractFeatureExtractor {

  public ProportionMissingValuesFeatureExtractor(
      DataSetInstance instance) {
    super(instance);
  }

  @Override
  public FeaturePair call() {
    final double numberOfMissingValues = extractProportionOfMissingValues();
    return new FeaturePair(SharedConstants.PROPORTION_OF_MISSING_VALUES,
        String.valueOf(numberOfMissingValues));
  }

  private double extractProportionOfMissingValues() {
    final Instances wekaInstance = instance.getWekaInstance();
    final Enumeration<Attribute> attributes = wekaInstance.enumerateAttributes();

    int missingValuesCount = 0;
    while (attributes.hasMoreElements()) {
      missingValuesCount += wekaInstance
          .attributeStats(attributes.nextElement().index()).missingCount;
    }

    // relative: #missingvalues/(#features*#instances)
    return (double) missingValuesCount / ((double) wekaInstance.numAttributes()
        * (double) wekaInstance.numInstances());
  }
}
