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

  private final FeaturePairs featurePairs = new FeaturePairs();

  private final ExecutorService executorService;
  private final DataSetInstance instance;


  public FeatureExtractor(ExecutorService executorService, DataSetInstance instance) {
    this.executorService = executorService;
    this.instance = instance;
  }


  private FeaturePairs extractAllFeatures() throws ExecutionException, InterruptedException {

    AbstractFeatureExtractor extractor = new NumClassFeatureExtractor(instance);
    AbstractFeatureExtractor extractor1 = new NumFeaturesFeatureExtractor(instance);


    Collection<Callable<FeaturePair>> list = new ArrayList<>();
    list.add(extractor);
    list.add(extractor1);


    List<Future<FeaturePair>> results = executorService.invokeAll(list);
    for (Future<FeaturePair> f : results) {
      featurePairs.addFeaturePair(f.get());
    }


    return featurePairs;
  }

  @Override
  public FeaturePairs call() throws Exception {
    return extractAllFeatures();
  }
}
