package at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors;

import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import at.hwl.machinelearning.ass3.metalearning.utils.FeaturePair;
import at.hwl.machinelearning.ass3.metalearning.utils.SharedConstants;
import java.util.Arrays;
import weka.core.Instances;

/**
 * <h4>About this class</h4>
 *
 * <p>Description</p>
 *
 * @author Daniel Fuevesi
 * @version 1.0.0
 * @since 1.0.0
 */
public class VarianceMeanFeatureExtractor extends AbstractFeatureExtractor {

  public VarianceMeanFeatureExtractor(
      final DataSetInstance instance) {
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
    final double[] filteredVariances = Arrays.stream(instances.variances()).filter(Double::isFinite)
        .toArray();
    final double sumOfVariances = Arrays.stream(filteredVariances).sum();
    final double mean = sumOfVariances / filteredVariances.length;
    return new FeaturePair(SharedConstants.VARIANCE_MEAN, String.valueOf(mean));
  }
}
