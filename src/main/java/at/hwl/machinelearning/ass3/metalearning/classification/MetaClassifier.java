package at.hwl.machinelearning.ass3.metalearning.classification;

import at.hwl.machinelearning.ass3.metalearning.classification.classifiers.IClassifiable;
import at.hwl.machinelearning.ass3.metalearning.utils.ClassificationResult;
import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import java.util.Random;
import java.util.concurrent.Callable;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

class MetaClassifier implements Callable<ClassificationResult> {

  private final IClassifiable classifier;
  private final String classifierName;
  private Instances wekaInstance;

  MetaClassifier(DataSetInstance instance, IClassifiable classifier) {

    this.classifier = classifier;
    wekaInstance = instance.getWekaInstance();
    this.classifierName = classifier.getName();
  }

  @Override
  public ClassificationResult call() throws Exception {
    wekaInstance = classifier.prepareInstance(wekaInstance);
    final double accuracy = classify();
    return new ClassificationResult(classifierName, accuracy);
  }

  private double classify() throws Exception {
    final Evaluation eval = new Evaluation(wekaInstance);
    final Classifier cls = classifier.getClassifier();
    eval.crossValidateModel(cls, wekaInstance, determineNumberOfFolds(), new Random(1517134737));
    return eval.pctCorrect();
  }

  private int determineNumberOfFolds() {
    if (wekaInstance.size() < 10) {
      return wekaInstance.size() - 1;
    } else {
      return 10;
    }
  }
}
