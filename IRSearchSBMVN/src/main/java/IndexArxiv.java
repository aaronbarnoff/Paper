import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.custom.CustomAnalyzer;
import org.apache.lucene.document.*;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.index.VectorSimilarityFunction;

import java.util.*;
import java.util.regex.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;

public class IndexArxiv {
    private static int counter = 0;
    private static String indexPath = "E:\\IR Project\\arxivPapers";
    private static String ignoreListPath = "E:\\IR Project\\Embeddings\\ignore_list.txt";
    private static String docsPath = "E:\\IR Project\\Crawler\\arxivCrawler2\\arxivPapers";
    private static String embeddingsPath = "E:\\IR Project\\Embeddings\\embeddings_indexed.json"; // Path to embeddings file
    private static String clusterPath = "E:\\IR Project\\Embeddings\\embeddings_clusters.json"; // Path to clusters
    private static Map<String, Integer> clusterAssignments = loadClusterAssignments(); // For clustering
    private static Map<Integer, Integer> ignorePairs = new HashMap<>();  // For near duplicates
    private static Set<Integer> ignoredDocs = new HashSet<>(); // For near duplicates

    public static void main(String[] args) throws Exception {
        //System.out.println(clusterAssignments);

        try (BufferedReader br = new BufferedReader(new FileReader(ignoreListPath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(" ");
                if (parts.length >= 2) {
                    int doc1 = Integer.parseInt(parts[0]);
                    int doc2 = Integer.parseInt(parts[1]);
                    ignorePairs.put(doc1, doc2);    // Add each pair of near duplicates to map
                }
            }
        } catch (IOException e) {
            System.err.println("Ignore list error: " + e.getMessage());
        }

        System.out.println("Indexing to directory '" + indexPath + " '...");
        Directory dir = FSDirectory.open(Paths.get(indexPath));

        String stopFileLocation = "E:\\IR Project";
        Path stopFilePath = Paths.get(stopFileLocation);
        Analyzer analyzer = CustomAnalyzer.builder(stopFilePath)
                .withTokenizer("standard")
                .addTokenFilter("lowercase")
                .addTokenFilter("stop", "ignoreCase", "true", "words", "myStopWordsEmpty.txt", "format", "wordset")
                .build();

        IndexWriterConfig iwc = new IndexWriterConfig(analyzer);
        IndexWriter writer = new IndexWriter(dir, iwc);

        ObjectMapper objectMapper = new ObjectMapper();
        Map<String, float[]> embeddingsMap = objectMapper.readValue(  // Extract embeddings so Lucene can use them
                new File(embeddingsPath), new TypeReference<>() {}
        );

        // Pass embeddingsMap to the indexDocs method
        indexDocs(writer, Paths.get(docsPath), embeddingsMap);
        writer.close();
    }

    /*
    public static String readBufferedReader(BufferedReader br) throws IOException {
        StringWriter writer = new StringWriter();
        char[] buffer = new char[1024];
        int numRead;
        while ((numRead = br.read(buffer)) != -1) {
            writer.write(buffer, 0, numRead);
        }
        return writer.toString();
    }
     */

    public static String removeLatex(String text) {
        text = text.replaceAll("\\\\[a-zA-Z]+(?:\\[[^\\]]*\\])?(?:\\{[^\\}]*\\})?", "");
        text = text.replaceAll("\\$.*?\\$", "");
        text = text.replaceAll("\\\\\\((.*?)\\\\\\)", "");
        Pattern multiLinePattern = Pattern.compile("\\\\begin\\{.*?\\}.*?\\\\end\\{.*?\\}", Pattern.DOTALL);
        Matcher matcher = multiLinePattern.matcher(text);
        text = matcher.replaceAll("");

        return text;
    }

    static void indexDoc(IndexWriter writer, Path file, Map<String, float[]> embeddingsMap) throws Exception {
        if (ignorePairs.containsKey(counter) || ignorePairs.containsValue(counter)) {
            int otherDoc = ignorePairs.getOrDefault(counter, -1);
            if (otherDoc == -1) {
                for (Map.Entry<Integer, Integer> entry : ignorePairs.entrySet()) {
                    if (entry.getValue() == counter) {
                        otherDoc = entry.getKey();
                        break;
                    }
                }
            }

            if (otherDoc != -1 && !ignoredDocs.contains(otherDoc)) { // SKip this one if the other isn't already ignored
                ignoredDocs.add(counter);
                System.out.println("Skipping doc: " + counter);
                counter++;
                return;
            }
        }
        FieldType fullField = new FieldType();
        fullField.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS);  // for my phrase search
        fullField.setStored(true);
        fullField.setTokenized(true);
        fullField.setOmitNorms(false);
        fullField.setStoreTermVectors(true);
        fullField.setStoreTermVectorPositions(true);
        fullField.setStoreTermVectorOffsets(true);

        InputStream stream = Files.newInputStream(file);
        BufferedReader br = new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8));
        Document doc = new Document();

        // This is specific to reading the scraped metadata from my arxiv scraper script
        String arxivID = removeLatex(br.readLine().substring("Arxiv_ID:".length()));    //Get Arxiv_ID:
        String published = removeLatex(br.readLine().substring("Published:".length()));
        String updated = removeLatex(br.readLine().substring("Updated:".length()));
        String title = removeLatex(br.readLine().substring("Title:".length()));
        String authors = removeLatex(br.readLine().substring("Authors:".length()));
        String doi = removeLatex(br.readLine().substring("DOI:".length()));
        String abstractx = removeLatex(br.readLine().substring("Abstract_Link:".length()));
        String pdfx = removeLatex(br.readLine().substring("PDF_Link:".length()));
        String journalx = removeLatex(br.readLine().substring("Journal_ref:".length()));
        String comments = removeLatex(br.readLine().substring("Comments:".length()));
        String primCat = removeLatex(br.readLine().substring("Primary_Category:".length()));
        String allCat = removeLatex(br.readLine().substring("All_Categories:".length()));
        String absSum = removeLatex(br.readLine().substring("Abstract:".length()));

        // For clustering
        String clusterID = String.valueOf(clusterAssignments.getOrDefault(String.valueOf(counter), -1)); // Counter is docID
        //System.out.println("doc: " + counter + ", clusterID: " + clusterID);

        /*
        System.out.println("arxivid:" + arxivID);
        System.out.println("published:" + published);
        System.out.println("updated:" + updated);
        System.out.println("title:" + title);
        System.out.println("authors:" + authors);
        System.out.println("doi:" + doi);
        System.out.println("abstractx:" + abstractx);
        System.out.println("pdfx:" + pdfx);
        System.out.println("journalx:" + journalx);
        System.out.println("comments:" + comments);
        System.out.println("primCat:" + primCat);
        System.out.println("allCat:" + allCat);
        System.out.println("absSum:" + absSum);
        // For clustering
        System.out.println("clusterID:" + clusterID);
        */

        /*
        doc.add(new StringField("path", file.toString(), Field.Store.YES));
        doc.add(new StringField("published", published, Field.Store.YES));
        doc.add(new StringField("updated", updated, Field.Store.YES));
        doc.add(new StringField("authors", authors, Field.Store.YES));
        doc.add(new StringField("doi", doi, Field.Store.YES));
        doc.add(new StringField("abstractx", abstractx, Field.Store.YES));
        doc.add(new StringField("pdfx", pdfx, Field.Store.YES));
        doc.add(new StringField("journalx", journalx, Field.Store.YES));
        doc.add(new StringField("comments", comments, Field.Store.YES));
        doc.add(new StringField("primCat", primCat, Field.Store.YES));
        doc.add(new StringField("allCat", allCat, Field.Store.YES));
        //doc.add(new Field("contents", readBufferedReader(br), fullField));
        */

        // I only need these fields at the moment
        doc.add(new StringField("arxivID", arxivID, Field.Store.YES));
        doc.add(new Field("title", title, fullField));
        doc.add(new Field("absSum", absSum, fullField));

        doc.add(new StringField("clusterID", clusterID, Field.Store.YES));        // For clustering

        String documentIndex = String.valueOf(counter); // Counter is docID
        doc.add(new StringField("docID", String.valueOf(counter), Field.Store.YES)); // Store docID in index

        // Add vector embeddings to lucene index
        float[] embedding = embeddingsMap.get(documentIndex);
        if (embedding != null) {
            KnnFloatVectorField knnField = new KnnFloatVectorField("embedding", embedding);
            doc.add(knnField);
            System.out.println("Indexing embedding for document index: " + documentIndex);
        } else {
            System.out.println("No embedding found for document index: " + documentIndex);
        }

        writer.addDocument(doc);
        counter++;
        if (counter % 1000 == 0) {
            System.out.println("Indexing " + counter + "-th file " + file.getFileName());
        }
    }

    // My archive folder is not nested
    static void indexDocs(final IndexWriter writer, Path path, Map<String, float[]> embeddingsMap) throws Exception {
        Files.walkFileTree(path, new SimpleFileVisitor<Path>() {
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) {
                try {
                    indexDoc(writer, file, embeddingsMap);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
                return FileVisitResult.CONTINUE;
            }
        });
    }

    private static Map<String, Integer> loadClusterAssignments() {
        ObjectMapper objectMapper = new ObjectMapper();
        try {
            // Load from the current directory
            return objectMapper.readValue(new File(clusterPath), new TypeReference<Map<String, Integer>>() {});
        } catch (IOException e) {
            e.printStackTrace();
            return new HashMap<>();  // Return an empty map in case of error
        }
    }
}