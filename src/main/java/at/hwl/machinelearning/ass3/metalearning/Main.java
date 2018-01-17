package at.hwl.machinelearning.ass3.metalearning;


import at.hwl.machinelearning.ass3.metalearning.datahandling.InstanceCreator;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class Main {


  public static void main(String[] args) throws Exception {
    final ExecutorService executorService = Executors.newFixedThreadPool(10);
    final InstanceCreator creator = new InstanceCreator("datasets/");

    final MetaLearner metaLearner = new MetaLearner(executorService, creator);
    metaLearner.learn();

    executorService.shutdown();
  }


}
