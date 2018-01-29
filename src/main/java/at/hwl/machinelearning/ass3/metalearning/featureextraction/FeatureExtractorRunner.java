package at.hwl.machinelearning.ass3.metalearning.featureextraction;

import at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors.DataSetNameExtractor;
import at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors.EntropyFeatureExtractor;
import at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors.NumClassFeatureExtractor;
import at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors.NumFeaturesFeatureExtractor;
import at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors.NumInstancesFeatureExtractor;
import at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors.ProportionMissingValuesFeatureExtractor;
import at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors.StandardDeviationFeatureExtractor;
import at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors.VarianceFeatureExtractor;
import at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors.descriptive.CorrelationMeanFeatureExtractor;
import at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors.descriptive.CorrelationStandardDeviationFeatureExtractor;
import at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors.descriptive.KurtosisMeanFeatureExtractor;
import at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors.descriptive.KurtosisStandardDeviationFeatureExtractor;
import at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors.descriptive.SkewnessMeanFeatureExtractor;
import at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors.descriptive.SkewnessStandardDeviationFeatureExtractor;
import at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors.descriptive.VarianceMeanFeatureExtractor;
import at.hwl.machinelearning.ass3.metalearning.featureextraction.extractors.descriptive.VarianceStandardDeviationFeatureExtractor;
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

public class FeatureExtractorRunner implements Callable<FeaturePairs> {

  private final Collection<Callable<FeaturePair>> featureExtractors = new ArrayList<>();
  private final FeaturePairs featurePairs = new FeaturePairs();


  private final ExecutorService executorService;
  private final DataSetInstance instance;


  public FeatureExtractorRunner(ExecutorService executorService, DataSetInstance instance) {
    this.executorService = executorService;
    this.instance = instance;
    this.initializeFeatureExtractors();
  }


  private void initializeFeatureExtractors() {
    featureExtractors.add(new DataSetNameExtractor(instance));
    featureExtractors.add(new NumClassFeatureExtractor(instance));
    featureExtractors.add(new NumFeaturesFeatureExtractor(instance));
    featureExtractors.add(new NumInstancesFeatureExtractor(instance));
    featureExtractors.add(new ProportionMissingValuesFeatureExtractor(instance));
    featureExtractors.add(new EntropyFeatureExtractor(instance));
    featureExtractors.add(new StandardDeviationFeatureExtractor(instance));
    featureExtractors.add(new VarianceFeatureExtractor(instance));
    featureExtractors.add(new VarianceMeanFeatureExtractor(instance));
    featureExtractors.add(new VarianceStandardDeviationFeatureExtractor(instance));
    featureExtractors.add(new SkewnessMeanFeatureExtractor(instance));
    featureExtractors.add(new SkewnessStandardDeviationFeatureExtractor(instance));
    featureExtractors.add(new KurtosisMeanFeatureExtractor(instance));
    featureExtractors.add(new KurtosisStandardDeviationFeatureExtractor(instance));
    featureExtractors.add(new CorrelationMeanFeatureExtractor(instance));
    featureExtractors.add(new CorrelationStandardDeviationFeatureExtractor(instance));
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
