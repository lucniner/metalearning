package at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors;

import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import at.hwl.machinelearning.ass3.metalearning.utils.FeaturePair;
import at.hwl.machinelearning.ass3.metalearning.utils.SharedConstants;
import java.util.Collections;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import weka.core.Attribute;
import weka.core.Instances;

/**
 * <h4>About this class</h4>
 *
 * <p>Description
 *
 * @author Daniel Fuevesi
 * @version 1.0.0
 * @since 1.0.0
 */
public class SkewnessStandardDeviationFeatureExtractor extends AbstractFeatureExtractor {

  public SkewnessStandardDeviationFeatureExtractor(final DataSetInstance instance) {
    super(instance);
  }

  /**
   * Computes a result, or throws an exception if unable to do so.
   *
   * @return computed result
   */
  @Override
  public FeaturePair call() {
    final Instances instances = instance.getWekaInstance();
    final DescriptiveStatistics skewnessList = new DescriptiveStatistics();
    Collections.list(instances.enumerateAttributes())
        .stream()
        .filter(Attribute::isNumeric)
        .map(attribute -> attributeSkewness(instances, attribute))
        .mapToDouble(DescriptiveStatistics::getSkewness)
        .filter(Double::isFinite)
        .forEach(skewnessList::addValue);
    final double std = skewnessList.getStandardDeviation();
    return new FeaturePair(SharedConstants.SKEWNESS_STD, String.valueOf(std));
  }

  private DescriptiveStatistics attributeSkewness(
      final Instances instances, final Attribute attribute) {
    final DescriptiveStatistics statistics = new DescriptiveStatistics();
    Collections.list(instances.enumerateInstances())
        .stream()
        .map(instance1 -> getDoubleValue(instance1.value(attribute)))
        .forEach(statistics::addValue);
    return statistics;
  }

  private double getDoubleValue(final double value) {
    if (!Double.isFinite(value)) {
      return 0;
    }
    return value;
  }
}
