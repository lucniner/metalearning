package at.hwl.machinelearning.ass3.metalearning.utils;

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
public class DescriptiveStatisticsUtil {

  public DescriptiveStatistics createFromAttribute(final Instances instances,
      final Attribute attribute) {
    final DescriptiveStatistics statistics = new DescriptiveStatistics();
    Collections.list(instances.enumerateInstances())
        .stream()
        .map(
            instance1 -> getDoubleValue(instance1.value(attribute)))
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
