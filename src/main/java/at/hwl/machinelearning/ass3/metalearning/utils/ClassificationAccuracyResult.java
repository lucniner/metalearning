package at.hwl.machinelearning.ass3.metalearning.utils;

import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;

public class ClassificationAccuracyResult {

  private final Map<String, Double> classifications = new HashMap<>();

  public void addResult(final String classifier, final double accuracy) {
    classifications.put(classifier, accuracy);
  }

  public void addResult(final ClassificationResult result) {
    this.addResult(result.getClassifier(), result.getAccuracy());
  }

  public String getBestClassifier() {

    Map.Entry<String, Double> min = Collections.min(
            classifications.entrySet(),
            Comparator.comparingDouble(Map.Entry::getValue)
    );

    return min.getKey();
  }

  public Map<String, Double> getAllClassificationResults() {
    return classifications;
  }


}
