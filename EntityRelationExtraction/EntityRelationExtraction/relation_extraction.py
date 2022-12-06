from .format_helper import ProgressDecorator
from spacy.util import filter_spans
from tqdm import tqdm
import pandas as pd
from .GPT_3 import get_result_df


class RelationshipExtractor:
    def __init__(self, model, tokenizer, nlp, matcher, api_key):
        self.model = model
        self.tokenizer = tokenizer
        self.nlp = nlp
        self.matcher = matcher
        self.api_key = api_key

    def get_paraphrased_textBetween(self, sentence, num_return_sequences=10, num_beams=10, length_penalty=0.8):
        '''
        Only perserve text between two entities
        Filiter the instances which the length of the in-between text
        is too short compared to the original sentence
        Parapharse the text with beam search (width = 10) using length penelty (0.9)
        Select the shortest output.
        Extract the verb from the output
        '''
        # tokenize the text to be form of a list of token IDs
        inputs = self.tokenizer([sentence], truncation=True, padding="longest", return_tensors="pt")
        # generate the paraphrased sentences
        outputs = self.model.generate(
            **inputs,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            length_penalty=length_penalty
        )
        res = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # get the shortest paraphrased sentence
        res.sort(key=lambda x: len(x))
        return res[0]

    def extract_verb(self, text_between):
        pattern = [{'POS': 'VERB', 'OP': '?'}, ]
        # Add pattern to matcher
        self.matcher.add("verb-phrases", [pattern], on_match=None)
        doc = self.nlp(text_between)
        # call the matcher to find matches
        matches = self.matcher(doc)
        if matches:
            spans = [doc[start:end] for _, start, end in matches]
            # return filter_spans(spans)[0]
            filtered = filter_spans(spans)
            if len(filtered) == 1:
                return filtered[0].text
        return None

    @ProgressDecorator(string="Extracting Entities Relationships")
    def extract_relation_batch(self, data, size):
        ori_sents = []
        entitesFirst = []
        entitiesSecond = []
        relation = []
        ori_sents_g = []
        entitesFirst_g = []
        entitiesSecond_g = []
        total = 0
        zero_or_more_verb_cnt = 0
        not_paraphrased = 0

        for i in tqdm(range(size)):
            ab = data[i]
            for ins in ab:
                total += 1

                ori_sent = ins['sentence']
                e1 = ins['first_entity']
                e2 = ins['second_entity']

                text_between = ins['text_between']
                # extract the only verb from intermediate text
                res = self.extract_verb(text_between)
                # if there is no verb or more than 1 verb, apply paraphraser and extract verb from the returned result
                if not res:
                    zero_or_more_verb_cnt += 1

                    res = self.get_paraphrased_textBetween(text_between)
                    res = self.extract_verb(res)
                if res:
                    ori_sents.append(ori_sent)
                    entitesFirst.append(e1)
                    entitiesSecond.append(e2)
                    relation.append(res)
                # if still no result, use gtp 3
                else:
                    not_paraphrased += 1

                    ori_sents_g.append(ori_sent)
                    entitesFirst_g.append(e1)
                    entitiesSecond_g.append(e2)
        df = pd.DataFrame(
            {'sentence': ori_sents, 'entity_1': entitesFirst, 'entity_2': entitiesSecond, 'relationship': relation})
        # print(relation)
        df_g = pd.DataFrame({'sentence': ori_sents_g, 'entity_1': entitesFirst_g, 'entity_2': entitiesSecond_g})

        df_g = get_result_df(df_g, self.api_key)
        df_g['relationship'] = df_g.relationship.apply(self.extract_verb)

        not_gpt_3 = df_g['relationship'].isna().sum()
        df_g.dropna(subset=['relationship'], inplace=True)

        print(f"Total number of rows processed: {total}")
        print(f"Number of rows direct verb approach processed: {total - zero_or_more_verb_cnt}")
        print(f"Number of rows paraphraser approach processed: {zero_or_more_verb_cnt - not_paraphrased}")
        print(f"Number of rows GPT 3 approach processed: {not_paraphrased - not_gpt_3}")
        print(f"Rows Dropped: {not_gpt_3}")

        return pd.concat([df, df_g], ignore_index=True)
