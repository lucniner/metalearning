package at.hwl.machinelearning.ass3.metalearning.utils;

public class FeaturePair {

  private final String featureName;
  private final Double featureValue;

  public FeaturePair(String featureName, Double featureValue) {
    this.featureName = featureName;
    this.featureValue = featureValue;
  }

  public String getFeatureName() {
    return featureName;
  }

  public Double getFeatureValue() {
    return featureValue;
  }

  @Override
  public String toString() {
    return "FeaturePair{" +
            "featureName='" + featureName + '\'' +
            ", featureValue=" + featureValue +
            '}';
  }
}
