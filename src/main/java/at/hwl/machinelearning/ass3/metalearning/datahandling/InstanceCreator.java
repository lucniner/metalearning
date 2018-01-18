package at.hwl.machinelearning.ass3.metalearning.datahandling;

import at.hwl.machinelearning.ass3.metalearning.exceptions.NoClassLabelFound;
import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstances;
import at.hwl.machinelearning.ass3.metalearning.utils.SharedConstants;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.File;
import java.net.URL;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

public class InstanceCreator {

  private final List<String> possibleClassLabels = Arrays.asList(SharedConstants.POSSIBLE_CLASS_LABELS);
  private final DataSetInstances instances = new DataSetInstances();

  private final String dataSetLocation;


  public InstanceCreator(String dataSetLocation) {
    this.dataSetLocation = dataSetLocation;
  }


  public DataSetInstances loadInstances() throws Exception {
    final List<String> dataSetsLocations = getAllDataSets();
    createDataSetInstances(dataSetsLocations);
    return instances;
  }

  private List<String> getAllDataSets() {
    final URL dataSets = this.getClass().getClassLoader().getResource(dataSetLocation);
    if (dataSets != null) {
      return getDataSetsFromURL(dataSets);
    } else {
      return Collections.emptyList();
    }
  }

  private List<String> getDataSetsFromURL(final URL url) {
    final String path = url.getPath();
    final File[] dataSetFiles = new File(path).listFiles();

    if (dataSetFiles != null) {
      return Arrays.stream(dataSetFiles).map(File::getPath).collect(Collectors.toList());
    } else {
      return Collections.emptyList();
    }
  }

  private void createDataSetInstances(final List<String> dataSetLocations) throws Exception {
    for (final String location : dataSetLocations) {
      final ConverterUtils.DataSource dataSource = new ConverterUtils.DataSource(location);
      final Instances instance = dataSource.getDataSet();
      final int classIndex = getClassIndex(instance);
      instance.setClassIndex(classIndex);
      instances.addDataSet(new DataSetInstance(location, instance));
    }

  }


  public DataSetInstance getSingleInstance() throws Exception {
    final ConverterUtils.DataSource dataSource = new ConverterUtils.DataSource(dataSetLocation);
    final Instances instance = dataSource.getDataSet();
    final int classIndex = getClassIndex(instance);
    instance.setClassIndex(classIndex);
    return new DataSetInstance(dataSetLocation, instance);
  }

  private int getClassIndex(final Instances instances) throws NoClassLabelFound {
    final List<Attribute> attributes = Collections.list(instances.enumerateAttributes());
    int i = 0;
    for (final Attribute attribute : attributes) {
      if (possibleClassLabels.contains(attribute.name().toLowerCase())) {
        return i;
      }

      i++;
    }
    throw new NoClassLabelFound("unable to identify class label in dataset");
  }


}
