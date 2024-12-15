package com.example.irsearchsbmvn;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.custom.CustomAnalyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.*;
import org.apache.lucene.index.*;
import org.apache.lucene.queries.spans.SpanMultiTermQueryWrapper;
import org.apache.lucene.queries.spans.SpanNearQuery;
import org.apache.lucene.queries.spans.SpanQuery;
import org.apache.lucene.queries.spans.SpanTermQuery;
import org.apache.lucene.queryparser.classic.MultiFieldQueryParser;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.*;
import org.apache.lucene.search.highlight.*;
import org.apache.lucene.search.highlight.Formatter;
import org.apache.lucene.search.similarities.ClassicSimilarity;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import org.springframework.web.bind.annotation.*;

import javax.annotation.PostConstruct;
import javax.naming.directory.SearchResult;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

@RestController
public class SearchController {
    // Make sure to update these
    String indexStr = "E:\\IR Project\\arxivPapers";
    String stopFileLocation = "E:\\IR Project";
    private static String embeddingsPath = "E:\\IR Project\\Embeddings\\embeddings_indexed.json"; // Path to embeddings file

    // Load the document embeddings into a map (fix for cluster similarity search)
    ObjectMapper objectMapper = new ObjectMapper();
    Map<String, float[]> embeddingsMap;
    {
        try {
            embeddingsMap = objectMapper.readValue(
                    new File(embeddingsPath), new TypeReference<Map<String, float[]>>() {}
            );
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
    private Directory indexDir;
    private Analyzer analyzer;
    Path stopFilePath = Paths.get(stopFileLocation);

    @PostConstruct
    public void init() throws Exception {

        analyzer = CustomAnalyzer.builder(stopFilePath)    //using custom analyzer to help with phrase searches
                .withTokenizer("standard")
                .addTokenFilter("lowercase")                        //i tried disabling lowercase to allow matching by case but not too helpful
                /*
                .addTokenFilter("ngram",                            //use ngram filter on index; didn't seem to help in my case
                        "minGramSize", "2",                         //avoid indexing single character words
                        "maxGramSize", "3",
                        "preserveOriginal","true")                  //keep the original token as well
                 */
                //.addTokenFilter("snowballPorter")                 //I didn't find stemmers were helpful for my phrase searching
                .addTokenFilter("stop",                             //remove stop words so i can do exact phrase matching
                        "ignoreCase", "true",                       //key-value pairs
                        "words", "myStopWordsEmpty.txt",
                        "format", "wordset")                        //single word per line
                .build();
        //analyzer = new StandardAnalyzer();
        try {
            indexDir = FSDirectory.open(Paths.get(indexStr));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @GetMapping("/combinedSearch")
    public Map<String, Object> combinedSearch(@RequestParam("query") String queryStr,
                                              @RequestParam(defaultValue = "0") int page,
                                              @RequestParam(defaultValue = "10") int pageSize) throws Exception {
        // Validate
        System.out.println("Query: " + queryStr);
        System.out.println("Current page: " + page);
        if (page < 0) {
            throw new IllegalArgumentException("Page number must be >= 0.");
        }
        if (pageSize <= 0) {
            throw new IllegalArgumentException("Page size must be > 0.");
        }
        List<SearchResult> results = new ArrayList<>(); // Results to be returned
        int totalHits = 0;
        int topK = Math.max(pageSize * (page + 1), 1);  // Want topK always atl 1

        // Get my MI bigram terms
        Map<String, List<String>> expandedQueryTerms = new HashMap<>();
        ObjectMapper mapper = new ObjectMapper();
        String expandedQueryPath = "E:\\IR Project\\Embeddings\\bigram_terms.json";
        try {
            expandedQueryTerms = mapper.readValue(new File(expandedQueryPath), HashMap.class);
        } catch (IOException e) {
            System.err.println("Error loading expanded query terms: " + e.getMessage());
        }

        try (DirectoryReader reader = DirectoryReader.open(indexDir)) {
            IndexSearcher searcher = new IndexSearcher(reader);

            // Regular search query (default Lucene BM25)
            MultiFieldQueryParser queryParser = new MultiFieldQueryParser(new String[]{"absSum", "title"}, analyzer);
            Query regularQuery = queryParser.parse(queryStr);

            // Now combine that query with all the bigrams from MI
            String[] queryTerms = queryStr.split("\\s+");
            List<String> bigrams = new ArrayList<>();
            for (int i = 0; i < queryTerms.length - 1; i++) { // Create bigrams out of user query
                bigrams.add(queryTerms[i].toLowerCase() + " " + queryTerms[i + 1].toLowerCase());
            }
            BooleanQuery.Builder expandedQueryBuilder = new BooleanQuery.Builder();
            expandedQueryBuilder.add(regularQuery, BooleanClause.Occur.SHOULD); // Add original query

            for (String bigram : bigrams) {
                if (expandedQueryTerms.containsKey(bigram)) {
                    List<String> relatedTerms = expandedQueryTerms.get(bigram);
                    for (String relatedTerm : relatedTerms) { // Add related terms to the query
                        Query relatedQuery = queryParser.parse("\"" + relatedTerm + "\""); // Phrase queries
                        expandedQueryBuilder.add(relatedQuery, BooleanClause.Occur.SHOULD);
                    }
                }
            }
            BooleanQuery expandedQuery = expandedQueryBuilder.build();

            TopDocs regularTopDocs = searcher.search(expandedQuery, topK);
            System.out.println("BM25 Expanded Search - Total hits found: " + regularTopDocs.totalHits.value);

            double maxRegScore = 0.0;
            for (ScoreDoc scoreDoc : regularTopDocs.scoreDocs) {
                maxRegScore = Math.max(maxRegScore, scoreDoc.score);
            }

            // Vector search
            float[] queryEmbedding = getEmbeddingForQuery(queryStr);
            KnnFloatVectorQuery vectorQuery = new KnnFloatVectorQuery("embedding", queryEmbedding, topK);
            TopDocs vectorTopDocs = searcher.search(vectorQuery, topK);
            System.out.println("Vector Search - Total hits found: " + vectorTopDocs.totalHits.value);

            // Combine results
            double weightFactor = 1.25;  // Testing to see if giving vector more/less weight is helpful
            Map<Integer, Double> combinedScores = new HashMap<>();

            for (ScoreDoc scoreDoc : regularTopDocs.scoreDocs) {
                double normalizedRegScore = (maxRegScore > 0) ? (scoreDoc.score / maxRegScore) : 0.0; // Normalize regular (BM25) since vector score is 0-1
                combinedScores.put(scoreDoc.doc, normalizedRegScore);
            }

            for (ScoreDoc scoreDoc : vectorTopDocs.scoreDocs) {
                double weightedVectorScore = scoreDoc.score * weightFactor;
                combinedScores.merge(scoreDoc.doc, weightedVectorScore, Double::sum);
            }

            // Reorder results by combined score
            List<Map.Entry<Integer, Double>> sortedResults = new ArrayList<>(combinedScores.entrySet());
            sortedResults.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));

            // Pagination, only want results on given page
            int start = page * pageSize;
            if (start >= sortedResults.size()) {
                throw new IllegalArgumentException("No results left");
            }
            int end = Math.min(start + pageSize, sortedResults.size());

            TermVectors termVectors = reader.termVectors();
            for (int i = start; i < end; i++) {
                Map.Entry<Integer, Double> entry = sortedResults.get(i);
                // v Highlighter v
                Document doc = searcher.doc(entry.getKey());
                Fields vector = termVectors.get(entry.getKey());
                QueryScorer s = new QueryScorer(regularQuery);
                Formatter f = new SimpleHTMLFormatter("<font style=\"color:blue\">","</font>");
                Highlighter h = new Highlighter(f, s);
                //h.setTextFragmenter(new SentenceFragmenter());
                TokenStream ts = TokenSources.getTermVectorTokenStreamOrNull("absSum", vector, h.getMaxDocCharsToAnalyze() - 1);
                String content = doc.get("absSum");
                String fragment = h.getBestFragments(ts, content, 3, "<font style=\"color:blue\"> [...] </font>");
                // ^ Highlighter ^

                results.add(new SearchResult(doc.get("title"), fragment,  doc.get("absSum"), doc.get("arxivID"), doc.get("arxivID"), Integer.parseInt(doc.get("docID"))));
                //System.out.println(new SearchResult(doc.get("title"), doc.get("absSum"), doc.get("url"), doc.get("arxivID")).getPdfUrl() );
                System.out.println("Combined Search - Doc ID: " + entry.getKey() + ", Combined Score: " + entry.getValue() +
                        ", Title: " + doc.get("title") + ", arxivID: "+ doc.get("arxivID") + ", docID: " + Integer.parseInt(doc.get("docID"))
                        + ", clusterID: " + doc.get("clusterID")
                );
            }

            totalHits = sortedResults.size();
        }

        // Return the results
        Map<String, Object> response = new HashMap<>();
        response.put("results", results);
        response.put("totalHits", totalHits);
        return response;
    }

    @GetMapping("/similarDocuments")
    public Map<String, Object> findSimilarDocuments(@RequestParam("docID") int docID,
                                                    @RequestParam(defaultValue = "5") int topK) throws IOException {
        List<SearchResult> results = new ArrayList<>();
        ObjectMapper objectMapper = new ObjectMapper();  // For converting JSON string to float[]

        try (DirectoryReader reader = DirectoryReader.open(indexDir)) {
            IndexSearcher searcher = new IndexSearcher(reader);

            Document originalDoc = searcher.doc(docID);
            String clusterID = originalDoc.get("clusterID"); // Get clusterID of target doc
            System.out.println("Original Doc: " + originalDoc.get("docID") + ", ClusterID: " + clusterID);

            Query clusterFilterQuery = new TermQuery(new Term("clusterID", clusterID)); // Restrict search to same cluster
            //System.out.println(originalDoc.get("embedding")); // Always null

            // I couldn't find a way to get the embedding from originalDoc through Lucene's index, it always seems to return null.
            // Instead I have to grab the embedding stored locally.
            float[] embedding = embeddingsMap.get(String.valueOf(docID));
            KnnFloatVectorQuery knnQuery = new KnnFloatVectorQuery("embedding", embedding, topK); // Create vector query out of target doc

            BooleanQuery combinedQuery = new BooleanQuery.Builder()
                    .add(knnQuery, BooleanClause.Occur.MUST)
                    .add(clusterFilterQuery, BooleanClause.Occur.FILTER)
                    .build();

            TopDocs topDocs = searcher.search(combinedQuery, topK+1); // Search the cluster of docs using target doc's embedding
            for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
                if (scoreDoc.doc != docID) {                    // Exclude target doc
                    Document similarDoc = searcher.doc(scoreDoc.doc);
                    //System.out.println("SimilarDoc: " + similarDoc.get("docID"));
                    results.add(new SearchResult(similarDoc.get("title"), null, similarDoc.get("absSum"),
                            similarDoc.get("url"), similarDoc.get("arxivID"), Integer.parseInt(similarDoc.get("docID")))); // No snippet to show
                }
            }
        }

        Map<String, Object> response = new HashMap<>();
        response.put("results", results);
        return response;
    }


    // For my python embedding script
    private float[] getEmbeddingForQuery(String query) {
        try {
            // Python helper script is listening on port 5000; it uses SBERT2 to embed the query and passes it back
            String url = "http://localhost:5000/generateEmbedding?text=" + URLEncoder.encode(query, "UTF-8");

            HttpClient client = HttpClient.newHttpClient();
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .GET()
                    .build();

            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString()); // Send query script

            ObjectMapper objectMapper = new ObjectMapper();
            Map<String, Object> responseMap = objectMapper.readValue(response.body(), Map.class);
            List<Double> embeddingList = (List<Double>) responseMap.get("embedding"); // Turn the embedding into a double

            // Convert to float[] so it can be used with lucene's knnFloatVector
            float[] embedding = new float[embeddingList.size()];
            for (int i = 0; i < embeddingList.size(); i++) {
                embedding[i] = embeddingList.get(i).floatValue();
            }

            return embedding;

        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Embedding failed: " + query, e);
        }
    }

    // This is passed back to the front end to display the results
    public static class SearchResult {
        private String title;
        private String snippet;
        private String absSum;
        private String url;
        private String arxivID;
        private int docID;

        public SearchResult(String title, String snippet, String absSum, String url, String arxivID, int docID) {
            this.title = title;
            this.snippet = snippet;
            this.absSum = absSum;
            this.docID = docID;
            this.url = url;
            if (arxivID != null && arxivID.contains("/")) {
                String[] parts = arxivID.split("/");
                this.arxivID = parts[parts.length - 1];
                //System.out.println(this.arxivID);
            } else {
                this.arxivID = arxivID;
            }
        }

        public String getTitle() {
            return title;
        }

        public String getSnippet() {
            return snippet;
        }

        public String getUrl() {
            return url;
        }

        public String getPdfUrl() {
            return "https://arxiv.org/pdf/" + arxivID;
        }

        public int getDocID() {
            return docID;
        }

        public String getAbsSum() {
            return absSum;
        }
    }
}
