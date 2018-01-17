package at.hwl.machinelearning.ass3.metalearning.classification;

import at.hwl.machinelearning.ass3.metalearning.utils.ClassificationResult;
import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

import java.util.concurrent.Callable;

public abstract class AbstractClassifier implements Callable<ClassificationResult> {

  final int trainTestSplitPercent;
  final Instances wekaInstace;

  AbstractClassifier(DataSetInstance instance, int trainTestSplitPercent) {

    this.trainTestSplitPercent = trainTestSplitPercent;
    wekaInstace = instance.getWekaInstance();
  }


  abstract Classifier getClassifier();

  double classify() throws Exception {
    final Instances training = getTrainingInstance();
    final Instances testing = getTestInstance();
    final Evaluation eval = new Evaluation(training);
    final Classifier cls = getClassifier();

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
