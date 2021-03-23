import weka.clusterers.ClusterEvaluation;
import weka.clusterers.Cobweb;
import weka.clusterers.SimpleKMeans;
import weka.clusterers.kvalid.SilhouetteIndex;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.File;
import java.io.IOException;
import java.math.BigDecimal;

public class Cluster {

    public static void main(String[] args) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("src/main/webapp/dataSet/dataSetVideogames.csv"));
        Instances dataSet = loader.getDataSet();

        EuclideanDistance df = new  EuclideanDistance();
        df.setAttributeIndices("first-last");

        SimpleKMeans simpleKMeans = new SimpleKMeans();
        simpleKMeans.setSeed(82);
        simpleKMeans.setNumClusters(29);

        int[] attributes = new int[1];
        attributes[0] = 0;

        Remove remove = new Remove();
        remove.setAttributeIndicesArray(attributes);
        remove.setInputFormat(dataSet);
        dataSet = Filter.useFilter(dataSet, remove);

        simpleKMeans.buildClusterer(dataSet);
        System.out.println(simpleKMeans);

        ClusterEvaluation eval = new ClusterEvaluation();
        eval.setClusterer(simpleKMeans);
        eval.evaluateClusterer(dataSet);
        System.out.println(eval.clusterResultsToString());

        /*

        //Per trovare numClusters, bestSeed e maxSilhouette

        int bestK = 0, bestSeed = 0;
        double max = 0;

        for(int i = 0; i <= 200; i++) {
            for(int j = 2; j <= 30; j++) {

                EuclideanDistance df2 = new  EuclideanDistance();
                df2.setAttributeIndices("first-last");
                SimpleKMeans cl2 = new SimpleKMeans();
                cl2.setSeed(i);
                cl2.setNumClusters(j);
                cl2.setDistanceFunction(df2);
                cl2.buildClusterer(dataSet);

                SilhouetteIndex si = new SilhouetteIndex();
                si.evaluate(cl2, cl2.getClusterCentroids(), dataSet, df2);
                BigDecimal valueSilhouette = (Double.isNaN(si.getGlobalSilhouette())) ? BigDecimal.valueOf(0) : BigDecimal.valueOf(si.getGlobalSilhouette());

                if ( valueSilhouette.compareTo(BigDecimal.valueOf(max)) == 1) {
                    max = Double.parseDouble(valueSilhouette.toString());
                    bestK = j;
                    bestSeed = i;
                }

            }
        }

        System.out.println("bestK: " + bestK + " bestSeed: " + bestSeed + " maxSilhouette: " + max);

         */

    }

}
