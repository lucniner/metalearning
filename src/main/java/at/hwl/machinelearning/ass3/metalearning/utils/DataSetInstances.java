package at.hwl.machinelearning.ass3.metalearning.utils;

import java.util.LinkedList;
import java.util.List;

public class DataSetInstances {

  private final List<DataSetInstance> instances = new LinkedList<>();

  public boolean addDataSet(final DataSetInstance instance) {
    return instances.add(instance);
  }

  public List<DataSetInstance> getAllInstances() {
    return instances;
  }
}
