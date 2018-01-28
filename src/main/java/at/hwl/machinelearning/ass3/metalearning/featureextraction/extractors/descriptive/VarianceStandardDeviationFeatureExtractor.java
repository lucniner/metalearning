package at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors.descriptive;

import at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors.AbstractFeatureExtractor;
import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import at.hwl.machinelearning.ass3.metalearning.utils.FeaturePair;
import at.hwl.machinelearning.ass3.metalearning.utils.SharedConstants;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

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
    final List<Double> variances = calculateVariances();
    final double mean = calculateVarianceMean(variances);
    final double numerator = calculateNumerator(variances, mean);
    final double standardDeviation =
        Math.sqrt(numerator / (variances.size() - 1)); // N - 1 as correction due to sample
    return new FeaturePair(SharedConstants.VARIANCE_STD, String.valueOf(standardDeviation));
  }

  private List<Double> calculateVariances() {
    return Arrays.stream(instance.getWekaInstance().variances())
        .filter(Double::isFinite)
        .boxed()
        .collect(Collectors.toList());
  }

  private double calculateVarianceMean(final List<Double> variances) {
    final double sumOfVariances = variances.stream().mapToDouble(Double::doubleValue).sum();
    return sumOfVariances / variances.size();
  }

  private double calculateNumerator(final List<Double> variances, final double mean) {
    return variances
        .stream()
        .map(variance -> Math.pow(Math.abs(variance - mean), 2))
        .mapToDouble(Double::doubleValue)
        .sum();
  }
}
