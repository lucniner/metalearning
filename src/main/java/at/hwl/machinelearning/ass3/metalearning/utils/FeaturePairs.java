package at.hwl.machinelearning.ass3.metalearning.utils;

import java.util.LinkedList;
import java.util.List;

public class FeaturePairs {

  private final List<FeaturePair> features = new LinkedList<>();


  public boolean addFeaturePair(final FeaturePair pair) {
    return features.add(pair);
  }

  public List<FeaturePair> getFeaturePairs() {
    return features;
  }
}
