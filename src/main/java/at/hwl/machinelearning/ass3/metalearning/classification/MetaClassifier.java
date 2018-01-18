package at.hwl.machinelearning.ass3.metalearning.classification;

import at.hwl.machinelearning.ass3.metalearning.classification.classifiers.IClassifyable;
import at.hwl.machinelearning.ass3.metalearning.utils.ClassificationResult;
import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

import java.util.Random;
import java.util.concurrent.Callable;

class MetaClassifier implements Callable<ClassificationResult> {

  private final IClassifyable classifier;
  private final String classifierName;
  private Instances wekaInstace;

  MetaClassifier(DataSetInstance instance, IClassifyable classifier) {

    this.classifier = classifier;
    wekaInstace = instance.getWekaInstance();
    this.classifierName = classifier.getName();
  }

  @Override
  public ClassificationResult call() throws Exception {
    wekaInstace = classifier.prepareInstance(wekaInstace);
    final double accuracy = classify();
    return new ClassificationResult(classifierName, accuracy);
  }

  private double classify() throws Exception {
    final Evaluation eval = new Evaluation(wekaInstace);
    final Classifier cls = classifier.getClassifier();
    eval.crossValidateModel(cls, wekaInstace, 10, new Random(1));
    return eval.pctCorrect();

  }
}
