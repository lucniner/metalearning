package at.hwl.machinelearning.ass3.metalearning.utils;

public class ClassificationResult {

  private final String classifier;
  private final double accuracy;

  public ClassificationResult(final String classifier, double accuracy) {
    this.classifier = classifier;
    this.accuracy = accuracy;
  }

  public String getClassifier() {
    return classifier;
  }

  public double getAccuracy() {
    return accuracy;
  }
}
