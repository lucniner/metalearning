package at.hwl.machinelearning.ass3.metalearning.datahandling;

import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstances;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.File;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class InstanceCreator {

  private final DataSetInstances instances = new DataSetInstances();

  private final String dataSetLocation;


  public InstanceCreator(String dataSetLocation) {
    this.dataSetLocation = dataSetLocation;
  }


  public DataSetInstances loadInstances() throws Exception {
    final List<String> dataSetsLocations = getAllDatasets();
    createDataSetInstances(dataSetsLocations);
    return instances;
  }

  private List<String> getAllDatasets() {
    final String path = this.getClass().getClassLoader().getResource("./datasets").getPath();
    final File[] dataSetFiles = new File(path).listFiles();
    return Arrays.stream(dataSetFiles).map(File::getPath).collect(Collectors.toList());
  }

  private void createDataSetInstances(final List<String> dasetLocations) throws Exception {
    for (final String location : dasetLocations) {
      final ConverterUtils.DataSource dataSource = new ConverterUtils.DataSource(dataSetLocation.concat(location));
      final Instances instance = dataSource.getDataSet();
      instances.addDataSet(new DataSetInstance(location, instance));
    }
  }
}
