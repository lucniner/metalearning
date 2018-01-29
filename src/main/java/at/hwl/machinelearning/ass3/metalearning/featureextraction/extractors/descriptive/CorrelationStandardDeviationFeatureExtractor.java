package at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors.descriptive;

import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import at.hwl.machinelearning.ass3.metalearning.utils.FeaturePair;
import at.hwl.machinelearning.ass3.metalearning.utils.SharedConstants;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
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
public class CorrelationStandardDeviationFeatureExtractor extends CorrelationMeanFeatureExtractor {

  public CorrelationStandardDeviationFeatureExtractor(final DataSetInstance instance) {
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
    return new FeaturePair(SharedConstants.CORRELATION_STD,
        String.valueOf(correlationMeans.getStandardDeviation()));
  }
}
