package at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors.descriptive;

import at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors.AbstractFeatureExtractor;
import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import at.hwl.machinelearning.ass3.metalearning.utils.FeaturePair;
import at.hwl.machinelearning.ass3.metalearning.utils.SharedConstants;
import java.util.Arrays;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

/**
 * <h4>About this class</h4>
 *
 * <p>Description
 *
 * @author Daniel Fuevesi
 * @version 1.0.0
 * @since 1.0.0
 */
public class VarianceStandardDeviationFeatureExtractor extends AbstractFeatureExtractor {

  public VarianceStandardDeviationFeatureExtractor(final DataSetInstance instance) {
    super(instance);
  }

  /**
   * Computes a result, or throws an exception if unable to do so.
   *
   * @return computed result
   */
  @Override
  public FeaturePair call() {
    final DescriptiveStatistics statistics = new DescriptiveStatistics();
    Arrays.stream(instance.getWekaInstance().variances())
        .filter(Double::isFinite)
        .forEach(statistics::addValue);
    return new FeaturePair(SharedConstants.VARIANCE_STD,
        String.valueOf(statistics.getStandardDeviation()));
  }
}
