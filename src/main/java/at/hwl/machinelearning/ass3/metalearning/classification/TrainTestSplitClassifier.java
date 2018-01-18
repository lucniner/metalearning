package at.hwl.machinelearning.ass3.metalearning.classification;

import at.hwl.machinelearning.ass3.metalearning.classification.classifiers.IClassifyable;
import at.hwl.machinelearning.ass3.metalearning.utils.ClassificationResult;
import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

import java.util.concurrent.Callable;

class TrainTestSplitClassifier implements Callable<ClassificationResult> {

  private final int trainTestSplitPercent;
  private final IClassifyable classifier;
  private final String classifierName;
  private Instances wekaInstace;


  TrainTestSplitClassifier(DataSetInstance instance, int trainTestSplitPercent, IClassifyable classifier) {

    this.trainTestSplitPercent = trainTestSplitPercent;
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
    final Instances training = getTrainingInstance();
    final Instances testing = getTestInstance();
    final Evaluation eval = new Evaluation(training);
    final Classifier cls = classifier.getClassifier();

    cls.buildClassifier(training);
    eval.evaluateModel(cls, testing);

    return eval.pctCorrect();

  }

  private Instances getTrainingInstance() {
    return new Instances(wekaInstace, 0, getTrainSize());
  }

  private Instances getTestInstance() {
    final int toCopy = wekaInstace.numInstances() - getTrainSize();
    return new Instances(wekaInstace, getTrainSize(), toCopy);
  }

  private int getTrainSize() {
    return (wekaInstace.numInstances() * trainTestSplitPercent / 100);
  }


}
