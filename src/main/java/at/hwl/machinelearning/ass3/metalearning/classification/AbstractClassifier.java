package at.hwl.machinelearning.ass3.metalearning.classification;

import at.hwl.machinelearning.ass3.metalearning.utils.ClassificationResult;
import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;

import java.util.concurrent.Callable;

public abstract class AbstractClassifier implements Callable<ClassificationResult> {

  final DataSetInstance instance;

  protected AbstractClassifier(DataSetInstance instance) {
    this.instance = instance;
  }
}
