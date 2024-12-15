import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.document.Document;
import org.apache.lucene.store.FSDirectory;
import java.nio.file.Paths;

public class ValidateIndex {
    public static void main(String[] args) throws Exception {
        String indexPath = "E:\\IR Project\\arxivPapers";
        IndexReader reader = DirectoryReader.open(FSDirectory.open(Paths.get(indexPath)));

        System.out.println("Total documents indexed: " + reader.maxDoc());

        for (int i = 0; i < Math.min(5, reader.maxDoc()); i++) { // Print some sample documents
            Document doc = reader.document(i);
            System.out.println("Doc " + i + ":");
            System.out.println("Title: " + doc.get("title"));
            System.out.println("Abstract: " + doc.get("absSum"));
            System.out.println("Arxiv ID: " + doc.get("arxivID"));
        }

        reader.close();
    }
}