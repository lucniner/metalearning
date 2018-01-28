package at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors;

import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import at.hwl.machinelearning.ass3.metalearning.utils.FeaturePair;
import at.hwl.machinelearning.ass3.metalearning.utils.SharedConstants;
import java.util.ArrayList;
import java.util.List;
import weka.core.Instances;

public class VarianceFeatureExtractor extends AbstractFeatureExtractor {

  public VarianceFeatureExtractor(
      DataSetInstance instance) {
    super(instance);
  }

  @Override
  public FeaturePair call() {
    final double maxTreeDepth = extractVariance();
    return new FeaturePair(SharedConstants.VARIANCE, String.valueOf(maxTreeDepth));
  }

  private double extractVariance() {
    final Instances wekaInstance = instance.getWekaInstance();
    final double[] variances = wekaInstance.variances();
    List<Double> variancesWithoutNaNs = new ArrayList<>();

    for (double variance : variances) {
      if (!Double.isNaN(variance)) {
        variancesWithoutNaNs.add(variance);
      }
    }

    double variancesSum = 0.0;
    for (Double variance : variancesWithoutNaNs) {
      variancesSum += variance;
    }
    return variancesSum / (double) variancesWithoutNaNs.size();
  }
}
