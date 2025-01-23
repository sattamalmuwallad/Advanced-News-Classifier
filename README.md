# Advanced News Classifier

## Overview

The **Advanced News Classifier** is a Java-based project developed as part of the Object-Oriented Programming (OOP) module at the University of Birmingham. This project leverages advanced machine learning techniques, such as GloVe embeddings and neural networks, to classify news articles into predefined categories. The project was an academic exercise designed to enhance technical skills in OOP, data processing, and machine learning.

---

## Features

- **Word Embedding with GloVe**: Utilizes pre-trained GloVe embeddings to capture semantic relationships between words.
- **Machine Learning Integration**: Implements a neural network using Deeplearning4J to classify news articles.
- **Text Preprocessing**: Includes text cleaning, lemmatization, and stop-word removal.
- **Custom Data Structures**: Designed classes like `Glove`, `NewsArticles`, and `ArticlesEmbedding` for modular and efficient data handling.
- **Efficient File Handling**: Processes large datasets with performance-optimized file operations.
- **Scalable Classification**: Supports dynamic classification of articles into multiple categories.

---

## Technical Skills Gained

- **Object-Oriented Programming**: Designed modular and reusable classes following Java OOP principles.
- **File I/O and Data Parsing**: Implemented efficient file reading and data parsing techniques for processing GloVe embeddings and news datasets.
- **Text Preprocessing**: Applied advanced NLP techniques such as lemmatization using CoreNLP and stop-word filtering.
- **Machine Learning Integration**: Gained experience using Deeplearning4J and NDArray4J for building and training neural networks.
- **Performance Optimization**: Focused on optimizing the execution time of critical tasks, such as embedding generation and data processing.

---

## How It Works

1. **GloVe File Processing**: Parses pre-trained GloVe embeddings to create word vectors.
2. **News Article Loading**: Reads and processes news articles stored in HTML format.
3. **Text Preprocessing Pipeline**:
   - Cleans and lemmatizes text.
   - Removes stop words.
   - Converts text to lowercase.
4. **Embedding Generation**: Creates document-level embeddings using word vectors.
5. **Neural Network Training**: Trains a neural network model to classify news articles.
6. **Prediction and Results**: Predicts categories for testing data and outputs classification results.

---

## Technologies Used

- **Java**: Core programming language.
- **Deeplearning4J**: Library for building and training neural networks.
- **NDArray4J**: Library for numerical and scientific computing in Java.
- **CoreNLP**: Natural Language Processing library for lemmatization.

---

## Project Structure

- **`Glove.java`**: Handles word embeddings.
- **`NewsArticles.java`**: Manages news article data.
- **`HtmlParser.java`**: Parses HTML files to extract relevant information.
- **`Toolkit.java`**: Provides utility methods for file handling and preprocessing.
- **`ArticlesEmbedding.java`**: Creates document-level embeddings for articles.
- **`AdvancedNewsClassifier.java`**: Main class for training and testing the news classifier.

---

## Key Learning Outcomes

This project allowed me to:
- Understand the importance of modularity and reusability in software design.
- Apply NLP techniques to real-world datasets.
- Gain hands-on experience with machine learning frameworks.
- Develop and debug a multi-component application with performance constraints.

---

## Future Enhancements

- Integrating transformer-based models like BERT for improved classification accuracy.
- Expanding the dataset to include more diverse categories.
- Developing a user-friendly GUI for easier interaction with the classifier.

---

For more details, you can explore the project [here](https://github.com/sattamalmuwallad/Advanced-News-Classifier).
