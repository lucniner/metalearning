package at.hwl.machinelearning.ass3.metalearning.utils;

import weka.core.Instances;

public class DataSetInstance {

  private final String dataSetLocation;
  private final Instances wekaInstance;

  public DataSetInstance(String dataSetLocation, Instances wekaInstance) {
    this.dataSetLocation = dataSetLocation;
    this.wekaInstance = wekaInstance;
  }

  public String getDataSetLocation() {
    return dataSetLocation;
  }

  public Instances getWekaInstance() {
    return wekaInstance;
  }
}
