package at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors;

import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import at.hwl.machinelearning.ass3.metalearning.utils.FeaturePair;
import at.hwl.machinelearning.ass3.metalearning.utils.SharedConstants;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Instances;

public class StandardDeviationFeatureExtractor extends AbstractFeatureExtractor {

  public StandardDeviationFeatureExtractor(
      DataSetInstance instance) {
    super(instance);
  }

  @Override
  public FeaturePair call() {
    final double standardDeviation = extractStandardDeviation();
    return new FeaturePair(SharedConstants.STANDARD_DEVIATION_MEAN, String.valueOf(standardDeviation));
  }

  private double extractStandardDeviation() {
    final Instances wekaInstance = instance.getWekaInstance();
    final Enumeration<Attribute> attributes = wekaInstance.enumerateAttributes();
    List<Double> stnDevs = new ArrayList<>();

    while (attributes.hasMoreElements()) {
      final Attribute attribute = attributes.nextElement();
      final AttributeStats attributeStats = wekaInstance.attributeStats(attribute.index());

      if (attribute.isNumeric()) {
        stnDevs.add(attributeStats.numericStats.stdDev);
      }
    }

    double sum = 0.0;
    for (Double stnDev : stnDevs) {
      sum += stnDev;
    }

    return sum / (double) stnDevs.size();
  }
}
