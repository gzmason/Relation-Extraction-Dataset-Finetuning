from setuptools import setup
import setuptools

setup(
   name='EntityRelationExtraction',
   version='0.0.1',
   author=""" Ruilin Liu  
              Zhifeng Zhang
              Zhiqing Yang
              Yutong (Jessie) Wang 
              Zhucheng Zhan
          """,
   author_email=""" rl3234@columbia.edu
                    zz2884@columbia.edu
                    zy2491@columbia.edu
                    yw3765@columbia.edu
                    zz2783@columbia.edu
          """,
   packages= setuptools.find_packages(),
   description="""Use a NER tagger to tag the entities in a set of documents.
For entity pairs within a certain proximity (e.g., same sentence or 20 tokens), pull out the text separating the entities.
Use abstractive summarization to compress these separating texts into just one or two tokens as the relationship between the two entities.
Semantically cluster these extreme summarizations, and identify a centroid term that is representative of all of them.
Mark the entity pairs in a given cluster with their corresponding centroid term, and a relation extraction dataset will be formed.
""",
)