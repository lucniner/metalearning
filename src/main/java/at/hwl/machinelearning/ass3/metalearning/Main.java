package at.hwl.machinelearning.ass3.metalearning;


import at.hwl.machinelearning.ass3.metalearning.datahandling.InstanceCreator;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class Main {

  private static final String TRAINING_DATA_SETS_LOCATION = "datasets/";
  private static final String META_EXTRACTION_RESULT_FILE = "metalearning_results.CSV";

  public static void main(String[] args) {

    final ExecutorService executorService = Executors.newFixedThreadPool(10);
    final InstanceCreator creator = new InstanceCreator(TRAINING_DATA_SETS_LOCATION);
    final InstanceCreator metaResultCreator = new InstanceCreator(META_EXTRACTION_RESULT_FILE);

    try {
      final MetaLearner metaLearner = new MetaLearner(executorService, creator, metaResultCreator, META_EXTRACTION_RESULT_FILE);
      metaLearner.learn();
    } catch (Exception e) {
      e.printStackTrace();
    } finally {
      executorService.shutdown();
    }

  }


}
