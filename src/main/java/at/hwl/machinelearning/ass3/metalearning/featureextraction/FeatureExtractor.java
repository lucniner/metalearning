package at.hwl.machinelearning.ass3.metalearning.featureextraction;

import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import at.hwl.machinelearning.ass3.metalearning.utils.FeaturePair;
import at.hwl.machinelearning.ass3.metalearning.utils.FeaturePairs;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

public class FeatureExtractor implements Callable<FeaturePairs> {

  private final Collection<Callable<FeaturePair>> featureExtractors = new ArrayList<>();
  private final FeaturePairs featurePairs = new FeaturePairs();


  private final ExecutorService executorService;
  private final DataSetInstance instance;


  public FeatureExtractor(ExecutorService executorService, DataSetInstance instance) {
    this.executorService = executorService;
    this.instance = instance;
    this.initializeFeatureExtractors();
  }


  private void initializeFeatureExtractors() {
    featureExtractors.add(new NumClassFeatureExtractor(instance));
    featureExtractors.add(new NumFeaturesFeatureExtractor(instance));
    featureExtractors.add(new NumInstancesFeatureExtractor(instance));
  }

  @Override
  public FeaturePairs call() throws Exception {
    return extractAllFeatures();
  }

  private FeaturePairs extractAllFeatures() throws ExecutionException, InterruptedException {
    final List<Future<FeaturePair>> results = executorService.invokeAll(featureExtractors);
    for (final Future<FeaturePair> f : results) {
      featurePairs.addFeaturePair(f.get());
    }
    return featurePairs;
  }


}
