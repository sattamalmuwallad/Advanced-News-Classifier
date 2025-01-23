package uob.oop;

import org.apache.commons.lang3.time.StopWatch;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class AdvancedNewsClassifier {
    public Toolkit myTK = null;
    public static List<NewsArticles> listNews = null;
    public static List<Glove> listGlove = null;
    public List<ArticlesEmbedding> listEmbedding = null;
    public MultiLayerNetwork myNeuralNetwork = null;

    public final int BATCHSIZE = 10;

    public int embeddingSize = 0;
    private static StopWatch mySW = new StopWatch();

    public AdvancedNewsClassifier() throws IOException {
        myTK = new Toolkit();
        myTK.loadGlove();
        listNews = myTK.loadNews();
        listGlove = createGloveList();
        listEmbedding = loadData();
    }

    public static void main(String[] args) throws Exception {
        mySW.start();
        AdvancedNewsClassifier myANC = new AdvancedNewsClassifier();

        myANC.embeddingSize = myANC.calculateEmbeddingSize(myANC.listEmbedding);
        myANC.populateEmbedding();
        myANC.myNeuralNetwork = myANC.buildNeuralNetwork(2);
        myANC.predictResult(myANC.listEmbedding);
        myANC.printResults();
        mySW.stop();
        System.out.println("Total elapsed time: " + mySW.getTime());
    }

    private boolean isStopWord(String word, String[] stopWords) {
        for (String stopWord : stopWords)
            if (stopWord.equalsIgnoreCase(word))
                return true;
        return false;
    }

    public List<Glove> createGloveList() {
        List<Glove> listResult = new ArrayList<>();
        //TODO Task 6.1 - 5 Marks
        List<String> vocabList = Toolkit.getListVocabulary();
        for (int i = 0; i < vocabList.size(); i++)
            if (!isStopWord(vocabList.get(i), Toolkit.STOPWORDS)) {
                double[] vecArr = Toolkit.listVectors.get(i);
                Vector vector = new Vector(vecArr);
                listResult.add(new Glove(vocabList.get(i), vector));
            }
        return listResult;
    }


    public static List<ArticlesEmbedding> loadData() {
        List<ArticlesEmbedding> listEmbedding = new ArrayList<>();
        for (NewsArticles news : listNews) {
            ArticlesEmbedding myAE = new ArticlesEmbedding(news.getNewsTitle(), news.getNewsContent(), news.getNewsType(), news.getNewsLabel());
            listEmbedding.add(myAE);
        }
        return listEmbedding;
    }

    public int calculateEmbeddingSize(List<ArticlesEmbedding> _listEmbedding) {
        int intMedian = -1;
        //TODO Task 6.2 - 5 Marks
        int totalDocumentCount = _listEmbedding.size();
        List<Integer> docLength = new ArrayList<>(totalDocumentCount);
        for (int i = 0; i < totalDocumentCount; i++) {
            ArticlesEmbedding embedding = _listEmbedding.get(i);
            String[] words = embedding.getNewsContent().split(" ");
            int count = 0;
            for (String word : words) {
                for (Glove glove : listGlove) {
                    if (glove.getVocabulary().equalsIgnoreCase(word)) {
                        count++;
                    }
                }
            }
            docLength.add(count);
        }
        quickSort(docLength, 0, docLength.size() - 1);
        if (docLength.size() % 2 == 0) {
            int i = (docLength.size() / 2) + 1;
            int j = docLength.size() / 2;
            intMedian = (docLength.get(i) + docLength.get(j)) / 2;
        } else {
            int i = (docLength.size() + 1) / 2;
            intMedian = docLength.get(i);
        }
        return intMedian;
    }

    private void quickSort(List<Integer> list, int low, int high) {
        if (low < high) {
            int e = quickSorthelper(list, low, high);
            quickSort(list, low, e - 1);
            quickSort(list, e + 1, high);
        }
    }

    private int quickSorthelper(List<Integer> list, int low, int high) {
        int p = list.get(high);
        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (list.get(j) < p) {
                i++;
                swap(list, i, j);
            }
        }
        swap(list, i + 1, high);
        return i + 1;
    }

    private void swap(List<Integer> list, int p, int q) {
        int temp = list.get(p);
        list.set(p, list.get(q));
        list.set(q, temp);
    }

    public void populateEmbedding() {
        //TODO Task 6.3 - 10 Marks
        for (ArticlesEmbedding articlesEmbedding : listEmbedding) {
            try {
                articlesEmbedding.getEmbedding();
            }catch (InvalidSizeException e){
                articlesEmbedding.setEmbeddingSize(embeddingSize);
            }catch (InvalidTextException e ){
                articlesEmbedding.getNewsContent();
            }catch (Exception e){
                //
            }
        }
    }

    public DataSetIterator populateRecordReaders(int _numberOfClasses) throws Exception {
        ListDataSetIterator myDataIterator = null;
        List<DataSet> listDS = new ArrayList<>();
        INDArray inputNDArray = null;
        INDArray outputNDArray = null;
        //TODO Task 6.4 - 8 Marks

        for (ArticlesEmbedding articlesEmbedding:listEmbedding) {
            inputNDArray = articlesEmbedding.getEmbedding();
            outputNDArray = Nd4j.zeros(1, _numberOfClasses);
            if (articlesEmbedding.getNewsType().equals(NewsArticles.DataType.Training)) {
                String label = articlesEmbedding.getNewsLabel();
                if (label.equalsIgnoreCase("1"))
                    outputNDArray.putScalar(0, 0, 1);
                 else
                    outputNDArray.putScalar(0, 1, 1);
                listDS.add(new DataSet(inputNDArray,outputNDArray));
            }
        }
        return new ListDataSetIterator(listDS, BATCHSIZE);
    }

    public MultiLayerNetwork buildNeuralNetwork(int _numOfClasses) throws Exception {
        DataSetIterator trainIter = populateRecordReaders(_numOfClasses);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(42)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(Adam.builder().learningRate(0.02).beta1(0.9).beta2(0.999).build())
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder().nIn(embeddingSize).nOut(15)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.HINGE)
                        .activation(Activation.SOFTMAX)
                        .nIn(15).nOut(_numOfClasses).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        for (int n = 0; n < 100; n++) {
            model.fit(trainIter);
            trainIter.reset();
        }
        return model;
    }

    public List<Integer> predictResult(List<ArticlesEmbedding> _listEmbedding) throws Exception {
        List<Integer> listResult = new ArrayList<>();
        //TODO Task 6.5 - 8 Marks
        for (ArticlesEmbedding articlesEmbedding : _listEmbedding) {
            if (articlesEmbedding.getNewsType() == NewsArticles.DataType.Testing) {
                int[] outputArr = myNeuralNetwork.predict(articlesEmbedding.getEmbedding());
                listResult.add(outputArr[0]);
                articlesEmbedding.setNewsLabel(String.valueOf(outputArr[0]));
            }
        }
        return listResult;
    }

    public void printResults() {
        //TODO Task 6.6 - 6.5 Marks

    }
}
