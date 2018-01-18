package at.hwl.machinelearning.ass3.metalearning.utils;

public class FeaturePair {

  private final String featureName;
  private final String featureValue;

  public FeaturePair(String featureName, String featureValue) {
    this.featureName = featureName;
    this.featureValue = featureValue;
  }

  public String getFeatureName() {
    return featureName;
  }

  public String getFeatureValue() {
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
