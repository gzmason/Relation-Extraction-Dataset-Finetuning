# Installation

Download this EntityRelationExtraction folder, install the package with
```
sudo pip install -e EntityRelationExtraction
```

# Apply for GPT-3 API Key

Since we will utilize OpenAI's GPT-3 in this pipeline, please refer to OpenAI's website https://openai.com/api/ to obtain an API Key.

# Sample Run
```
from EntityRelationExtraction.driver import Driver
d = Driver('API Key From Previous Step')
d.prepareDataSet(abstracts)
```
"abstracts" is a list of raw text inputs. 
Results that contains entity pairs and their corresponding relations will be produced at ./result/result.csv as indicated at the end of the run.
