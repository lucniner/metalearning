package at.hwl.machinelearning.ass3.metalearning.utils;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import weka.core.Attribute;
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
public class DescriptiveStatisticsUtil {

  public DescriptiveStatistics createFromAttribute(
      final Instances instances, final Attribute attribute) {
    final DescriptiveStatistics statistics = new DescriptiveStatistics();
    Collections.list(instances.enumerateInstances())
        .stream()
        .map(instance1 -> getDoubleValue(instance1.value(attribute)))
        .forEach(statistics::addValue);
    return statistics;
  }

  public Map<Integer, double[]> getAttributeValuesMap(final Instances instances) {
    final Map<Integer, double[]> result = initializeMap(instances);
    final AtomicInteger counter = new AtomicInteger(-1);
    final Set<Integer> attributes = result.keySet();
    Collections.list(instances.enumerateInstances())
        .forEach(
            instance -> {
              counter.incrementAndGet();
              attributes.forEach(
                  attribute ->
                      result.get(attribute)[counter.get()] =
                          getDoubleValue(instance.value(attribute)));
            });
    return result;
  }

  private Map<Integer, double[]> initializeMap(final Instances instances) {
    final Map<Integer, double[]> result = new HashMap<>();
    Collections.list(instances.enumerateAttributes())
        .stream()
        .filter(Attribute::isNumeric)
        .forEach(attribute -> result.put(attribute.index(), new double[instances.size()]));
    return result;
  }

  private double getDoubleValue(final double value) {
    if (!Double.isFinite(value)) {
      return 0;
    }
    return value;
  }
}
