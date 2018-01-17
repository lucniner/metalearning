package at.hwl.machinelearning.ass3.metalearning.utils;

import java.util.LinkedList;
import java.util.List;

public class FeaturePairs {

  private final List<FeaturePair> featurePairs = new LinkedList();


  public boolean addFeaturePair(final FeaturePair pair) {
    return featurePairs.add(pair);
  }

  public List<FeaturePair> getFeaturePairs() {
    return featurePairs;
  }
}
