import inflect
from OpenNRE import opennre
from .relation_extraction import RelationshipExtractor
from transformers import PegasusForConditionalGeneration, PegasusTokenizerFast
import spacy
from spacy.matcher import Matcher
import pickle
import gensim
from .relation_clustering import Clustering
from .entity_extractor import EntityExtractor
import os
from .format_helper import printHeader, printEnd


class Driver:
    def __init__(self, api_key):
        # Define the tokenizer and model for paraphraser
        self.model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase")
        self.tokenizer = PegasusTokenizerFast.from_pretrained("tuner007/pegasus_paraphrase")
        # Define the nlp core for Spacy
        self.nlp = spacy.load('en_core_sci_lg')
        # Instantiate a Matcher instance to match the verb
        self.matcher = Matcher(self.nlp.vocab)
        # Define OpenAI api key
        self.key = api_key
        # Load the simple dict
        with open("simple_dict.txt", "rb") as file:
            self.simple_dict = pickle.load(file)
        # Load an inflect_converter
        self.inflect_converter = inflect.engine()
        # Load an OpenNRE model
        self.opennre_model = opennre.get_model('wiki80_cnn_softmax')
        # Load Word2vec pretrained embeddings
        if os.path.exists("word2vec.vectors.npy") and os.path.exists("word2vec"):
            self.word2vec_model = gensim.models.KeyedVectors.load('word2vec', mmap='r')
        else:
            self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
                "GoogleNews-vectors-negative300.bin.gz", binary=True)
            self.word2vec_model.init_sims(replace=True)
            self.word2vec_model.save('word2vec')

    def prepareDataSet(self, df):
        # Entities extraction
        ee = EntityExtractor(self.nlp, self.opennre_model, self.simple_dict, self.inflect_converter)
        data = ee.extract_entities(df)

        # Relation extraction
        re = RelationshipExtractor(self.model, self.tokenizer, self.nlp, self.matcher, self.key)
        df_relation_extracted = re.extract_relation_batch(data, len(data))

        # Clustering relationships
        c = Clustering(self.word2vec_model, False, 100)
        df_final_result = c.get_cluster_word(df_relation_extracted, 0.7)
        df_final_result = df_final_result.drop(["relationship", "relationship_lem", "embedding", "cluster"], axis=1)

        if not os.path.exists("result"):
            os.mkdir("result/")
        df_final_result.to_csv("result/result.csv")

        print("Done! Result dataset can be found in result/result.csv")
