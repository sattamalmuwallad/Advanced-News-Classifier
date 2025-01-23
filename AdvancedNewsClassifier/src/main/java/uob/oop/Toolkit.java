package uob.oop;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;

public class Toolkit {
    public static List<String> listVocabulary = null;
    public static List<double[]> listVectors = null;
    private static final String FILENAME_GLOVE = "glove.6B.50d_Reduced.csv";

    public static final String[] STOPWORDS = {"a", "able", "about", "across", "after", "all", "almost", "also", "am", "among", "an", "and", "any", "are", "as", "at", "be", "because", "been", "but", "by", "can", "cannot", "could", "dear", "did", "do", "does", "either", "else", "ever", "every", "for", "from", "get", "got", "had", "has", "have", "he", "her", "hers", "him", "his", "how", "however", "i", "if", "in", "into", "is", "it", "its", "just", "least", "let", "like", "likely", "may", "me", "might", "most", "must", "my", "neither", "no", "nor", "not", "of", "off", "often", "on", "only", "or", "other", "our", "own", "rather", "said", "say", "says", "she", "should", "since", "so", "some", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "tis", "to", "too", "twas", "us", "wants", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would", "yet", "you", "your"};

    public void loadGlove() throws IOException {
        try {
            File path = getFileFromResource(FILENAME_GLOVE);
            BufferedReader myReader = new BufferedReader(new FileReader(path));
            //TODO Task 4.1 - 5 marks
            String line;
            listVocabulary = new ArrayList<>();
            listVectors = new ArrayList<>();
            while ((line = myReader.readLine()) != null) {
                String[] data = line.split(",");
                listVocabulary.add(data[0]);
                double[] vector = new double[data.length - 1];
                for (int i = 1; i < data.length; i++)
                    vector[i - 1] = Double.parseDouble(data[i]);
                listVectors.add(vector);
            }
        } catch (Exception e) {
            e.getMessage();
        }
    }

    private static File getFileFromResource(String fileName) throws URISyntaxException {
        ClassLoader classLoader = Toolkit.class.getClassLoader();
        URL resource = classLoader.getResource(fileName);
        if (resource == null) {
            throw new IllegalArgumentException(fileName);
        } else {
            return new File(resource.toURI());
        }
    }

    public List<NewsArticles> loadNews() {
        List<NewsArticles> listNews = new ArrayList<>();
        //TODO Task 4.2 - 5 Marks
        try {
            File path = getFileFromResource("News");
            String line;
            File[] files = path.listFiles(((dir, name) -> name.toLowerCase().endsWith(".htm")));
            if (files != null)
                for (File file : files) {
                    StringBuilder SB = new StringBuilder();
                    BufferedReader myReader = new BufferedReader(new FileReader(file));
                    while ((line = myReader.readLine()) != null)
                        SB.append(line);
                    String content = SB.toString();
                    NewsArticles news = new NewsArticles(HtmlParser.getNewsTitle(content), HtmlParser.getNewsContent(content), HtmlParser.getDataType(content), HtmlParser.getLabel(content));
                    listNews.add(news);
                }
        } catch (Exception e) {
            e.getMessage();
        }
        listNews.sort(Comparator.comparing(NewsArticles::getNewsTitle));
        return listNews;
    }

    public static List<String> getListVocabulary() {
        return listVocabulary;
    }

    public static List<double[]> getlistVectors() {
        return listVectors;
    }
}
