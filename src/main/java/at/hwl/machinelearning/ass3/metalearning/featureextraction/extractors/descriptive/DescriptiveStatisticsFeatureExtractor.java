package at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors.descriptive;

import at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors.AbstractFeatureExtractor;
import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import at.hwl.machinelearning.ass3.metalearning.utils.DescriptiveStatisticsUtil;

/**
 * <h4>About this class</h4>
 *
 * <p>Description</p>
 *
 * @author Daniel Fuevesi
 * @version 1.0.0
 * @since 1.0.0
 */
abstract class DescriptiveStatisticsFeatureExtractor extends AbstractFeatureExtractor {

  final DescriptiveStatisticsUtil descriptiveStatisticsUtil;

  DescriptiveStatisticsFeatureExtractor(
      final DataSetInstance instance) {
    super(instance);
    descriptiveStatisticsUtil = new DescriptiveStatisticsUtil();
  }
}
