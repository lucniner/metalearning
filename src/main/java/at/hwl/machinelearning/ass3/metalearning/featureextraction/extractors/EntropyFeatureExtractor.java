package at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors;

import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import at.hwl.machinelearning.ass3.metalearning.utils.FeaturePair;
import at.hwl.machinelearning.ass3.metalearning.utils.SharedConstants;
import java.util.HashMap;
import java.util.Map;
import weka.core.Instance;
import weka.core.Instances;

public class EntropyFeatureExtractor extends AbstractFeatureExtractor {

  public EntropyFeatureExtractor(
      DataSetInstance instance) {
    super(instance);
  }

  @Override
  public FeaturePair call() {
    final double entropy = extractEntropy();
    return new FeaturePair(SharedConstants.ENTROPY, String.valueOf(entropy));
  }

  private double extractEntropy() {
    final Instances wekaInstance = instance.getWekaInstance();
    final int numOfInstances = wekaInstance.numInstances();

    Map<Object, Integer> instancePerClass = new HashMap<>();

    for (int i = 0; i < wekaInstance.numInstances(); i++) {
      final Instance instance = wekaInstance.get(i);
      final double[] instanceValues = instance.toDoubleArray();
      final double classValue = instanceValues[wekaInstance.classIndex()];

      instancePerClass.merge(classValue, 1, (a, b) -> a + b);
    }

    double entropy = 0.0;
    for (Integer cnt : instancePerClass.values()) {
      final double classProbability = (double) cnt / (double) numOfInstances;
      entropy -= classProbability * (Math.log(classProbability) / Math.log(2));
    }

    return entropy;
  }
}
