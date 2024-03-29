package at.hwl.machinelearning.ass3.metalearning.utils;

public class SharedConstants {

  public static final String NUMBER_OF_CLASSES = "number_of_classes";
  public static final String NUMBER_OF_FEATURES = "number_of_features";
  public static final String NUMBER_OF_INSTANCES = "number_of_instances";
  public static final String PROPORTION_OF_MISSING_VALUES = "proportion_of_missing_values";
  public static final String ENTROPY = "entropy";
  public static final String STANDARD_DEVIATION_MEAN = "standard_deviation_mean";
  public static final String CLASS_LABEL = "class";
  public static final String VARIANCE_MEAN = "variance_mean";
  public static final String VARIANCE_STD = "variance_std";
  public static final String SKEWNESS_MEAN = "skewness_mean";
  public static final String SKEWNESS_STD = "skewness_std";
  public static final String KURTOSIS_MEAN = "kurtosis_mean";
  public static final String KURTOSIS_STD = "kurtosis_std";
  public static final String CORRELATION_MEAN = "correlation_mean";
  public static final String CORRELATION_STD = "correlation_std";
  public static final String REP_TREE_SIZE = "rep_tree_size";

  public static final String[] POSSIBLE_CLASS_LABELS = {
      CLASS_LABEL,
          "label"
  };

  private SharedConstants() {
  }
}
