package at.hwl.machinelearning.ass3.metalearning.datahandling;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class MetaResultWriter {

  private static final String LINE_BREAK = "\n";

  private final Map<String, String> accuracyResults;
  private final Map<String, List<String>> featureResults;
  private final List<String> featureNames;

  public MetaResultWriter(
      Map<String, String> accuracyResults,
      Map<String, List<String>> featureResults,
      List<String> featureNames) {
    this.accuracyResults = accuracyResults;
    this.featureResults = featureResults;
    this.featureNames = featureNames;
  }

  public void writeResultsToFile(final String fileLocation) throws IOException {
    handleResultFile(fileLocation);
  }

  private void handleResultFile(final String fileLocation) throws IOException {
    final File file = new File(fileLocation);
    try (final BufferedWriter writer = new BufferedWriter(new FileWriter(file))) {
      writeHeader(writer);
      writeResults(writer);
    }
  }

  private void writeHeader(final BufferedWriter writer) throws IOException {
    final String header = String.join(",", featureNames).concat(",class");
    writer.write(header);
    writer.write(LINE_BREAK);
  }

  private void writeResults(final BufferedWriter writer) throws IOException {
    for (final Map.Entry<String, List<String>> entry : featureResults.entrySet()) {
      final String classifier = accuracyResults.get(entry.getKey());
      final String featureValues = getFeatureValuesAsString(entry.getValue());
      final String result = featureValues.concat(",").concat(classifier);
      writer.write(result);
      writer.write(LINE_BREAK);
    }
  }

  private String getFeatureValuesAsString(final List<String> featurevalues) {
    final List<String> featureValue = new ArrayList<>(featurevalues);
    return String.join(",", featureValue);
  }
}
