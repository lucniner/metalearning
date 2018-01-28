package at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors.descriptive;

import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import at.hwl.machinelearning.ass3.metalearning.utils.FeaturePair;
import at.hwl.machinelearning.ass3.metalearning.utils.SharedConstants;
import java.util.Collections;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import weka.core.Attribute;
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
public class KurtosisMeanFeatureExtractor extends DescriptiveStatisticsFeatureExtractor {

  public KurtosisMeanFeatureExtractor(
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
    final AtomicInteger counter = new AtomicInteger(0);
    final double kurtosisSum =
        Collections.list(instances.enumerateAttributes())
            .stream()
            .filter(Attribute::isNumeric)
            .map(attribute -> descriptiveStatisticsUtil.createFromAttribute(instances, attribute))
            .mapToDouble(DescriptiveStatistics::getKurtosis)
            .peek(operand -> counter.incrementAndGet())
            .filter(Double::isFinite)
            .sum();
    if (counter.get() == 0) {
      counter.incrementAndGet();
    }
    final double kurtosisMean = kurtosisSum / counter.get();
    return new FeaturePair(SharedConstants.KURTOSIS_MEAN, String.valueOf(kurtosisMean));
  }
}
