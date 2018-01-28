package at.hwl.machinelearning.ass3.metalearning.classification.classifiers;

import weka.classifiers.Classifier;
import weka.core.Instances;

public interface IClassifiable {

  Classifier getClassifier();

  Instances prepareInstance(final Instances instances);

  String getName();
}
