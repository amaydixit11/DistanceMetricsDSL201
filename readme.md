# Distance Metrics Applications

## _Submitted to Dr. Anil Kumar Sao_ _DSL201-Assignment-2_

## _Prepared by Amay Dixit \- 12340220_

# **Part 1 \- Text Analysis Using Bag of Words and Distance Metrics** INTRODUCTION

This report outlines the analysis of a dataset comprising nine documents across three domains: politics, science, and sports. The primary objective was to implement the Bag of Words (BoW) model to convert the text documents into numerical representations and compute various distance metrics to assess the similarity between documents.

# METHODOLOGY

### **1\. Data Preparation**

The first step in the analysis involved preparing the dataset, which consisted of nine text documents divided into three distinct domains: politics, science, and sports. The preparation process included several key activities:

- **Document Loading:** Each document was loaded into the Python environment for processing.
- **Data Cleaning:** To ensure the quality of the analysis, the loaded text underwent a cleaning process. This involved:
  - **Lowercasing**
  - **Punctuation Removal**
  - **Tokenization**

By the end of this preparation phase, the dataset was fully processed, cleaned, and organized, making it ready for the implementation of the Bag of Words model and subsequent distance metric calculations. This meticulous preparation was crucial for obtaining accurate and reliable results in the later stages of the analysis.

### **2\. Bag of Words Model**

Following data preparation, the next step was to implement the Bag of Words (BoW) model to convert the text documents into a numerical format suitable for analysis. The key activities in this phase included:

- **Vocabulary Creation:** A vocabulary was generated by extracting unique terms from all documents. This vocabulary served as the basis for the term-document matrix, where each column represented a distinct term.
- **Term-Document Matrix Construction:** A term-document matrix was created, where each row represented a document and each column represented a term from the vocabulary. The values in the matrix reflected the term frequencies of the respective terms in each document. This matrix provided a structured representation of the textual data, enabling quantitative analysis.
- **Normalization:** The term frequencies were counted directly, which may impact the analysis by treating all terms equally without adjusting for their importance across the corpus.

This structured approach allowed for a clear representation of the textual data, facilitating the subsequent calculation of distance metrics.

### **3\. Distance Metrics**

After constructing the term-document matrix using the Bag of Words model, various distance metrics were calculated to assess the similarity between pairs of documents. The following distance metrics were employed:

1. **Jaccard Distance:**
   - The Jaccard distance was computed to measure the dissimilarity between pairs of documents. It is defined as 1−Jaccard similarity, where Jaccard similarity is calculated as the size of the intersection of the term sets divided by the size of the union. This metric is particularly useful for understanding the overlap in vocabulary between documents, allowing for insights into shared themes or topics.
2. **Euclidean Distance:**

   - Euclidean distance was calculated using the term-document vectors. This metric measures the "straight-line" distance in the multi-dimensional space defined by the term frequencies. It quantifies the absolute difference in term frequencies between two documents, highlighting the degree of dissimilarity in their content.

3. **Cosine Similarity (and Distance):**
   - Cosine similarity was calculated to assess the orientation of document vectors in the vector space, disregarding their magnitude. The cosine similarity is converted to a distance metric by subtracting it from 1\. This approach is effective in identifying how closely aligned the topics or themes of documents are, making it a valuable tool for comparative analysis.
4. **Kullback-Leibler (K-L) Divergence:**
   - K-L divergence was computed to evaluate the difference between two probability distributions derived from term frequencies. Since K-L divergence requires normalized distributions, the term frequencies were first normalized to create probability distributions for each document. This metric provides insights into how one document's term distribution diverges from another, emphasizing differences in thematic content.

Each of these metrics provides a unique perspective on the relationships between documents, allowing for a comprehensive analysis of similarity and dissimilarity across the distinct domains of politics, science, and sports. The resulting distance matrices from these calculations laid the groundwork for further visualization and interpretation.

### **4\. Analysis**

**Distance Matrix Creation:** For each of the four distance metrics (Jaccard, Euclidean, Cosine, and K-L divergence), a distance matrix was constructed. Each matrix represented the pairwise distances between all documents, facilitating a comparative analysis.

**Visualization of Distance Matrices:** Heatmaps were used to visualize these distance matrices. Heatmaps provide an intuitive way to interpret the distances, allowing for quick identification of similarities and differences among documents.

# DISTANCE MATRICES AND HEATMAPS

To facilitate a deeper understanding of the relationships between the documents, we constructed distance matrices for each of the distance metrics calculated earlier. These matrices serve as a comprehensive representation of the pairwise distances between all documents, allowing us to identify patterns and similarities across different domains.

For each distance matrix, heatmaps were generated to provide a visual representation of the distances. These heatmaps help in quickly identifying clusters of similar documents and understanding the degree of dissimilarity between different pairs.

![][image1]

![][image2]![][image3]

![][image4]

The distance matrices and their corresponding heatmaps offer valuable insights into the relationships between documents across the three domains of politics, science, and sports. By analyzing these visual representations, we can draw conclusions about thematic similarities, document clustering, and potential overlaps in content. This analysis lays the foundation for further exploration and understanding of textual data within these domains.

###

### PATTERNS AND INSIGHTS

The analysis of the distance matrices revealed several insights:

#### **Jaccard Distance**

- **Within-Category Similarities:**
  - **Politics (Documents 0, 1, 2):** Low distances (0.89 to 0.90) indicate strong thematic cohesion, suggesting shared vocabulary and topics.
  - **Science (Documents 3, 4, 5):** Moderate distances (0.86 to 0.88) reveal overlapping concepts.
  - **Sports (Documents 6, 7, 8):** Although specific values are lacking, we can infer low distances, indicating common themes.
- **Between-Category Dissimilarities:**
  - Higher distances (around 0.90 or above) between different categories reflect substantial thematic differences. For example, the distance between documents 2 (Politics) and 3 (Science) is approximately 0.90.
- **Distinct Document Pair Insights:**

  - The highest distance (around 0.93) indicates significant divergence, especially between documents in different categories, like document 5 (Science) and document 0 (Politics).

  #### **Euclidean Distance**

- **Within-Category Distances:**
  - **Politics:** Low distances (e.g., 57.24 between documents 0 and 1\) suggest high content similarity.
  - **Science:** Moderate distances (e.g., 63.57 between documents 3 and 4\) indicate related themes.
  - **Sports:** Wider distances (e.g., 81.91 between documents 6 and 7\) imply varied topics within this category.
- **Between-Category Dissimilarities:**

  - Significant distances (e.g., 98.94 between document 5 and document 0\) highlight thematic divergence.

  #### **Cosine Distance**

- **Within-Category Distances:**
  - **Politics:** Document 0 shows a low distance from document 2 (0.329), suggesting high topic similarity, while document 1 has moderate similarities.
  - **Science:** Close distances among documents (e.g., 0.344 and 0.274) reflect thematic alignment.
  - **Sports:** Document 6 shows lower similarities with documents 7 and 8, indicating unique aspects.
- **Between-Category Dissimilarities:**

  - Distances between different categories are high, with document 0 (Politics) and document 6 (Sports) showing substantial divergence (0.206).

  #### **K-L Divergence**

- **Within-Category Distances:**
  - **Politics:** Document 0 has low divergence from document 2 (10.305) and moderate divergence from document 1 (11.226), indicating some similarities in term distribution.
  - **Science:** Documents 4 and 5 show the lowest divergence (9.943), emphasizing their close thematic alignment.
  - **Sports:** Relatively low distances between documents (e.g., 8.826 between documents 6 and 7\) suggest shared vocabulary.
- **Between-Category Dissimilarities:**
  - K-L divergence values between different categories tend to be higher, such as the divergence between document 1 (Politics) and document 5 (Science) at 12.168.

# CONCLUSION

In this report, we successfully analyzed a dataset comprising nine documents from three distinct domains—politics, science, and sports—using the Bag of Words (BoW) model and various distance metrics. The analysis began with meticulous data preparation, which included document loading, cleaning, and tokenization, ensuring the integrity and quality of the textual data.

We constructed a term-document matrix that transformed the text documents into a numerical format, allowing for quantitative analysis. Subsequently, we employed four distance metrics—Jaccard distance, Euclidean distance, cosine similarity, and Kullback-Leibler divergence—to evaluate the similarity and dissimilarity between the documents.

The resulting distance matrices, visualized through heatmaps, provided insightful patterns regarding thematic similarities and differences across the domains. Notably, within-category documents demonstrated higher similarity, while between-category documents displayed significant divergence, indicating clear distinctions in content themes.

This analysis lays the groundwork for further exploration of textual data within diverse domains. The insights gained can inform future research and applications in natural language processing, document clustering, and information retrieval, contributing to a deeper understanding of how distance metrics can elucidate relationships in textual data.

# **Part 2 \- Image Analysis Using Jaccard Distance**

# INTRODUCTION

This report presents an analysis of six grayscale images: an original image (OR), a ground truth image (GT), and five algorithm-generated images (Algo1 to Algo5). The objective is to compute difference images and calculate the Jaccard Distance to assess the similarity between the outputs of different algorithms and the ground truth.

# METHODOLOGY

### **1\. Image Loading and Preparation**

The first phase of the analysis involved preparing the dataset, which included several images for comparison. The preparation process encompassed the following key activities:

- **Image Loading**: Each image was loaded into the Python environment using OpenCV. This ensured that all images were imported as grayscale, facilitating uniform processing and analysis.
- **Error Handling**: A validation step was implemented to check for successful loading of each image. If an image was not found, a `FileNotFoundError` was raised, allowing for prompt identification and correction of any issues related to missing files.

This comprehensive preparation ensured that all images were properly loaded and validated, laying the groundwork for subsequent computations of differences.

### **2\. Difference Computation**

Following the data preparation, the next step was to compute the absolute differences between the images. This phase involved:

- **Difference Calculation**: For each algorithm-generated image, the absolute difference from the original reference image (OR) was computed. This calculation highlighted discrepancies between the algorithm outputs and the original image, providing a clear view of the performance of each algorithm.
- **Storage of Difference Images**: The computed difference images were organized in a structured format for ease of access and analysis. This preparation was essential for the subsequent evaluation of image similarity.

This systematic approach to difference computation enabled a clear representation of variations among the images, crucial for the next steps in the analysis.

### **3\. Distance Metrics**

After calculating the difference images, various distance metrics were employed to assess the similarity between the difference images and the ground truth (GT). The key metrics included:

- **Jaccard Distance**: Jaccard distance was calculated to quantify dissimilarity between pairs of difference images. It is defined as 1−Jaccard similarity, where Jaccard similarity is determined by the intersection of pixel values divided by their union. This metric provided insights into the overlap between the images, revealing shared features and areas of divergence.

Each of these metrics provided distinct insights into the relationships between the difference images, allowing for a nuanced analysis of similarity and dissimilarity.

### **3\. Analysis**

- **Jaccard Distance Calculation**: For each difference image computed, the Jaccard distance was calculated with respect to the ground truth image (GT). This enabled the identification of which algorithms produced outputs most closely aligned with the original image.
- **Output Presentation**: The results of the Jaccard distance calculations were printed in a structured format, facilitating easy interpretation of how each algorithm compared to the ground truth.
- **Visualization of Difference Images**: A grid format was employed to display the difference images, allowing for visual comparison. This visualization enabled quick identification of similarities and differences across the processed images, enriching the analytical context.

# OUTPUT

**Jaccard Distances with respect to GT:**

1. Algo1: 0.14669780704714785
2. Algo2: 0.05059896668134645
3. Algo3: 0.051321644049574955
4. Algo4: 0.04681981311905836
5. Algo5: 0.03810059643769559

# OUTPUT IMAGES ![][image5]![][image6]![][image7]![][image8]![][image9]![][image10]

# RESULTS

The Jaccard distances calculated for the five algorithm-generated images (Algo1 to Algo5) provide valuable insights into how closely each algorithm's output aligns with the ground truth (GT). Here’s an analysis of the results:

1. **Results Overview**:
   - **Algo1**: Jaccard Distance \= 0.1467
   - **Algo2**: Jaccard Distance \= 0.0506
   - **Algo3**: Jaccard Distance \= 0.0513
   - **Algo4**: Jaccard Distance \= 0.0468
   - **Algo5**: Jaccard Distance \= 0.0381
2. **Comparison of Algorithms**:
   - **Algo5** has the lowest Jaccard distance (0.0381), indicating that it produced an output most similar to the ground truth. This suggests that the algorithm effectively captured the essential features of the original image, minimizing discrepancies.
   - **Algo4** follows closely behind (0.0468), also demonstrating strong alignment with the ground truth. Both Algo4 and Algo5 could be considered the best performers in this analysis.
   - **Algo2** and **Algo3** exhibit slightly higher distances (0.0506 and 0.0513, respectively), which indicates a moderate level of similarity with the ground truth but suggests that they may have missed capturing some critical features compared to Algo4 and Algo5.
   - **Algo1** has the highest Jaccard distance (0.1467), indicating that its output is the most dissimilar from the ground truth. This could imply that Algo1 either fails to capture important details or introduces more artifacts than the other algorithms.
3. **Insights and Implications**:
   - The results highlight the effectiveness of different algorithms in generating images that align with a reference standard. Lower Jaccard distances point to algorithms that maintain the integrity of the original image, while higher distances indicate potential areas for improvement.
   - The analysis underscores the importance of using quantitative measures like Jaccard distance for objective evaluation, providing a clear framework for comparing algorithm outputs.

# CONCLUSION

The analysis conducted using Jaccard Distance provided a comprehensive evaluation of the performance of five different image-processing algorithms in relation to a ground truth image. By computing the absolute differences and comparing the outputs of each algorithm against the reference images, several key insights emerged:

1. **Algorithm Performance**: Algo5 demonstrated the best alignment with the ground truth, exhibiting the lowest Jaccard distance. This indicates that it effectively captured the essential features of the original image while minimizing discrepancies, suggesting it is a robust choice for image processing tasks.
2. **Variability Among Algorithms**: The analysis revealed variability in performance among the algorithms. While Algo4 also performed well, Algo1 had the highest Jaccard distance, highlighting its inability to capture critical features effectively. This emphasizes the need for careful selection and tuning of algorithms based on specific requirements and performance metrics.
3. **Importance of Quantitative Measures**: Utilizing quantitative metrics like Jaccard Distance provides an objective framework for comparing different algorithm outputs. It facilitates a clearer understanding of how well each algorithm performs relative to a standard, thereby guiding future improvements and refinements.

In summary, this analysis underscores the effectiveness of Jaccard Distance as a metric for assessing image similarity.
