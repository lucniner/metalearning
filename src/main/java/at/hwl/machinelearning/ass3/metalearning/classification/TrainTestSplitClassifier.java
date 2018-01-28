package at.hwl.machinelearning.ass3.metalearning.classification;

import at.hwl.machinelearning.ass3.metalearning.classification.classifiers.IClassifiable;
import at.hwl.machinelearning.ass3.metalearning.utils.ClassificationResult;
import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import java.util.concurrent.Callable;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

class TrainTestSplitClassifier implements Callable<ClassificationResult> {

  private final int trainTestSplitPercent;
  private final IClassifiable classifier;
  private final String classifierName;
  private Instances wekaInstance;

  TrainTestSplitClassifier(
      DataSetInstance instance, int trainTestSplitPercent, IClassifiable classifier) {

    this.trainTestSplitPercent = trainTestSplitPercent;
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
    final Instances training = getTrainingInstance();
    final Instances testing = getTestInstance();
    final Evaluation eval = new Evaluation(training);
    final Classifier cls = classifier.getClassifier();

    cls.buildClassifier(training);
    eval.evaluateModel(cls, testing);

    return eval.pctCorrect();
  }

  private Instances getTrainingInstance() {
    return new Instances(wekaInstance, 0, getTrainSize());
  }

  private Instances getTestInstance() {
    final int toCopy = wekaInstance.numInstances() - getTrainSize();
    return new Instances(wekaInstance, getTrainSize(), toCopy);
  }

  private int getTrainSize() {
    return (wekaInstance.numInstances() * trainTestSplitPercent / 100);
  }
}
