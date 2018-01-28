package at.hwl.machinelearning.ass3.metalearning.datahandling;

import at.hwl.machinelearning.ass3.metalearning.exceptions.NoClassLabelFound;
import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstance;
import at.hwl.machinelearning.ass3.metalearning.utils.DataSetInstances;
import at.hwl.machinelearning.ass3.metalearning.utils.SharedConstants;
import java.io.File;
import java.net.URL;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class InstanceCreator {

  private final List<String> possibleClassLabels =
      Arrays.asList(SharedConstants.POSSIBLE_CLASS_LABELS);
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
      final List<String> result =
          Arrays.stream(dataSetFiles).map(File::getPath).collect(Collectors.toList());
      return result.subList(0, 3);
    } else {
      return Collections.emptyList();
    }
  }

  private void createDataSetInstances(final List<String> dataSetLocations) throws Exception {
    for (final String location : dataSetLocations) {
      final ConverterUtils.DataSource dataSource = new ConverterUtils.DataSource(location);
      Instances instance = dataSource.getDataSet();
      instance = filterIdAttribute(instance);
      final String instanceName = getInstanceName(location);
      final int classIndex = getClassIndex(instance);
      instance.setClassIndex(classIndex);
      instance.setRelationName(instanceName);
      instances.addDataSet(new DataSetInstance(location, instance));
    }
  }

  public DataSetInstance getSingleInstance() throws Exception {
    final ConverterUtils.DataSource dataSource = new ConverterUtils.DataSource(dataSetLocation);
    final Instances instance = dataSource.getDataSet();
    final String instanceName = getInstanceName(dataSetLocation);
    final int classIndex = getClassIndex(instance);
    instance.setClassIndex(classIndex);
    instance.setRelationName(instanceName);
    return new DataSetInstance(dataSetLocation, instance);
  }

  private Instances filterIdAttribute(final Instances instances) {
    return Collections.list(instances.enumerateAttributes())
        .stream()
        .filter(attribute -> attribute.name().toUpperCase().contains("ID"))
        .findFirst()
        .map(
            attribute -> {
              final Remove remove = new Remove();
              final String[] options = new String[2];
              options[0] = "-R";
              options[1] = String.valueOf(attribute.index() + 1);
              return applyFilter(instances, remove, options);
            })
        .orElse(instances);
  }

  private Instances applyFilter(
      final Instances instances, final Filter filter, final String[] options) {
    try {
      filter.setOptions(options);
      filter.setInputFormat(instances);
      return Filter.useFilter(instances, filter);
    } catch (Exception e) {
      e.printStackTrace();
      return instances;
    }
  }

  private String getInstanceName(final String location) {
    return location.substring(location.lastIndexOf('/') + 1, location.lastIndexOf('.'));
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
    throw new NoClassLabelFound("Unable to identify class label in data set");
  }
}
