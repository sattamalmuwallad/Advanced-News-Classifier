package uob.oop;

import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.pipeline.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.util.Properties;


public class ArticlesEmbedding extends NewsArticles {
    private int intSize = -1;
    private String processedText = "";

    private INDArray newsEmbedding = Nd4j.create(0);

    public ArticlesEmbedding(String _title, String _content, NewsArticles.DataType _type, String _label) {
        //TODO Task 5.1 - 1 Mark
        super(_title, _content, _type, _label);
    }

    public void setEmbeddingSize(int _size) {
        //TODO Task 5.2 - 0.5 Marks
        this.intSize = _size;
    }

    public int getEmbeddingSize() {
        return intSize;
    }

    @Override
    public String getNewsContent() {
        //TODO Task 5.3 - 10 Marks
        if (processedText.equals("")) {
            String content = super.getNewsContent();
            String cleaned = textCleaning(content);
            String lemma = lemmatizedText(cleaned);
            processedText = removeStopWords(lemma, Toolkit.STOPWORDS);
            return processedText.toLowerCase().trim();
        }
        return processedText.toLowerCase();
    }

    private String removeStopWords(String text, String[] stopWords) {
        StanfordCoreNLP pipeline = createPipeline();
        StringBuilder sb = new StringBuilder();
        String[] words = text.split(" ");
        for (String word : words)
            if (!isStopWord(word, stopWords, pipeline))
                sb.append(word).append(" ");
        return sb.toString().trim();
    }

    private boolean isStopWord(String word, String[] stopWords, StanfordCoreNLP pipeline) {
        CoreDocument document = new CoreDocument(word);
        pipeline.annotate(document);
        String lemma = document.tokens().get(0).lemma();
        for (String stopWord : stopWords)
            if (lemma.equalsIgnoreCase(stopWord))
                return true;
        return false;
    }

    private String lemmatizedText(String text) {
        StanfordCoreNLP pipeline = createPipeline();
        CoreDocument document = new CoreDocument(text);
        pipeline.annotate(document);
        StringBuilder sb = new StringBuilder();
        for (CoreLabel token : document.tokens())
            sb.append(token.lemma()).append(" ");
        return sb.toString().trim();
    }

    private StanfordCoreNLP createPipeline() {
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,pos,lemma");
        return new StanfordCoreNLP(props);
    }

    public INDArray getEmbedding() throws Exception {
        //TODO Task 5.4 - 20 Marks
        int vectorSize = AdvancedNewsClassifier.listGlove.get(0).getVector().getVectorSize();
        intSize = getEmbeddingSize();
        if (intSize == -1)
            throw new InvalidSizeException("Invalid size");
        if (processedText.isEmpty())
            throw new InvalidTextException("Invalid Text");
        if (newsEmbedding.isEmpty()) {
            newsEmbedding = Nd4j.zeros(intSize, vectorSize);
            String[] words = processedText.split(" ");
            int count = 0;
            int i = 0;
            for (String word : words) {
                if (count >= intSize)
                    break;
                for (Glove glove : AdvancedNewsClassifier.listGlove)
                    if (word.equals(glove.getVocabulary())) {
                        newsEmbedding.putRow(count, Nd4j.create(glove.getVector().getAllElements()));
                        count++;
                        break;
                    }
            }
            if (count == 0)
                throw new InvalidTextException("Invalid Text");
        }
        return Nd4j.vstack(newsEmbedding.mean(1));
    }

    /***
     * Clean the given (_content) text by removing all the characters that are not 'a'-'z', '0'-'9' and white space.
     * @param _content Text that need to be cleaned.
     * @return The cleaned text.
     */
    private static String textCleaning(String _content) {
        StringBuilder sbContent = new StringBuilder();

        for (char c : _content.toLowerCase().toCharArray()) {
            if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || Character.isWhitespace(c)) {
                sbContent.append(c);
            }
        }

        return sbContent.toString().trim();
    }
}
