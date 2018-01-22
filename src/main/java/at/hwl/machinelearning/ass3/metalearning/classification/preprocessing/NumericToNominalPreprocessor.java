package at.hwl.machinelearning.ass3.metalearning.classification.preprocessing;

import at.hwl.machinelearning.ass3.metalearning.utils.SharedConstants;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class NumericToNominalPreprocessor {

  private final Map<Double, String> classMapping = new HashMap<>();
  private final Instances instances;


  public NumericToNominalPreprocessor(Instances instances) {
    this.instances = instances;
    this.generateNominalClasses();
  }

  private void generateNominalClasses() {
    final double[] distinctClassValues = instances.attributeToDoubleArray(instances.classIndex());
    for (Double value : distinctClassValues) {
      classMapping.putIfAbsent(value, SharedConstants.CLASS_LABEL.concat(String.valueOf(value)));
    }
  }

  public Instances preprocess() {
    addNominalAttributeToInstance();
    addNominalValuesToInstances();
    updateClassIndex();
    return instances;
  }

  private void addNominalAttributeToInstance() {
    final List<String> nominalClassLabels = new ArrayList<>(classMapping.values());
    instances.insertAttributeAt(new Attribute("nominal_class", nominalClassLabels), instances.numAttributes());
  }

  private void addNominalValuesToInstances() {
    for (int i = 0; i < instances.numInstances(); i++) {
      final Instance instance = instances.get(i);
      final double[] instanceValues = instance.toDoubleArray();
      final double classValue = instanceValues[instances.classIndex()];
      instances.instance(i).setValue(instances.numAttributes() - 1, classMapping.get(classValue));
    }
  }

  private void updateClassIndex() {
    instances.setClassIndex(instances.numAttributes() - 1);

  }
}
