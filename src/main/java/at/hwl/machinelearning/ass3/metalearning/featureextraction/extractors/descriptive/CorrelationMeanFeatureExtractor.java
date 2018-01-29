package at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors.descriptive;

import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import at.hwl.machinelearning.ass3.metalearning.utils.FeaturePair;
import at.hwl.machinelearning.ass3.metalearning.utils.SharedConstants;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import weka.core.Instances;

/**
 *
 *
 * <h4>About this class</h4>
 *
 * <p>Description
 *
 * @author Daniel Fuevesi
 * @version 1.0.0
 * @since 1.0.0
 */
public class CorrelationMeanFeatureExtractor extends DescriptiveStatisticsFeatureExtractor {

  private final PearsonsCorrelation correlation = new PearsonsCorrelation();

  public CorrelationMeanFeatureExtractor(final DataSetInstance instance) {
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
    final DescriptiveStatistics correlationMeans = new DescriptiveStatistics();
    final Map<Integer, double[]> attributeValuesMap =
        descriptiveStatisticsUtil.getAttributeValuesMap(instances);
    final List<Integer> keys = new ArrayList<>(attributeValuesMap.keySet());
    keys.stream()
        .mapToDouble(
            attribute -> calculateMeanCorrelationOfAttribute(attribute, attributeValuesMap))
        .forEach(correlationMeans::addValue);
    return new FeaturePair(
        SharedConstants.CORRELATION_MEAN, String.valueOf(correlationMeans.getMean()));
  }

  double calculateMeanCorrelationOfAttribute(
      final int attributeIndex, final Map<Integer, double[]> attributeValuesMap) {
    final double[] values = attributeValuesMap.get(attributeIndex);
    final DescriptiveStatistics statistics = new DescriptiveStatistics();
    attributeValuesMap
        .values()
        .stream()
        .mapToDouble(otherColumn -> correlation(values, otherColumn))
        .forEach(statistics::addValue);
    return statistics.getMean();
  }

  private double correlation(double[] attribute1, double[] attribute2) {
    final double result = correlation.correlation(attribute1, attribute2);
    if (!Double.isFinite(result)) {
      return 0;
    }
    return result;
  }
}
