package at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors.descriptive;

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
 * <p>Description</p>
 *
 * @author Daniel Fuevesi
 * @version 1.0.0
 * @since 1.0.0
 */
public class KurtosisStandardDeviationFeatureExtractor extends
    DescriptiveStatisticsFeatureExtractor {

  public KurtosisStandardDeviationFeatureExtractor(
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
    final DescriptiveStatistics kurtosisValues = new DescriptiveStatistics();
    Collections.list(instances.enumerateAttributes())
        .stream()
        .filter(Attribute::isNumeric)
        .map(attribute -> descriptiveStatisticsUtil.createFromAttribute(instances, attribute))
        .mapToDouble(DescriptiveStatistics::getKurtosis)
        .filter(Double::isFinite)
        .forEach(kurtosisValues::addValue);
    final double std = kurtosisValues.getStandardDeviation();
    return new FeaturePair(SharedConstants.KURTOSIS_STD, String.valueOf(std));
  }
}
