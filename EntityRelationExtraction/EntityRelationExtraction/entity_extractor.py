from tqdm import tqdm
from .format_helper import ProgressDecorator

class EntityExtractor:
    def __init__(self, nlp, opennre_model, simple_dict, inflect_converter):
        self.nlp = nlp
        self.opennre_model = opennre_model
        self.simple_dict = simple_dict
        self.inflect_converter = inflect_converter

    def entity_extraction(self, text_input):
        doc = self.nlp(text_input)
        ignore_list = []
        text_entity_list = []
        for sentence in doc.sents:
            sentence_entity_list = []
            entity_start = -1
            entity_end = -1
            and_keyword_connecting = False
            prev_adp = False
            parenthesis_count = 0

            for token in sentence:
                if token.ent_type != 0 and token.tag_ == 'JJ':
                    if entity_start == -1:  # Entering a new entity
                        entity_start = token.idx
                elif token.ent_type != 0 and token.pos_ == 'NOUN':
                    if entity_start == -1:  # Entering a new entity
                        entity_start = token.idx
                    entity_end = token.idx + len(token)  # Possible to end an entity here
                elif token.ent_type != 0 and not prev_adp:
                    entity_start = -1
                    entity_end = -1
                    prev_adp = False
                elif token.text == "of":  # ignore the previous entity
                    entity_start = -1
                    entity_end = -1
                elif token.pos_ == 'ADP':  # only take entities after proposition
                    prev_adp = True
                elif token.text == "and" and entity_end == -1:  # if not yet reaches the end of an entity
                    continue
                else:
                    # checks if an entity has been formed
                    if entity_start != -1 and entity_end != -1:
                        if and_keyword_connecting and sentence_entity_list and (
                                entity_end - sentence_entity_list[-1]['end'] <= 30):
                            # if close enough with previous entity and connected by "AND"
                            sentence_entity_list[-1]['end'] = entity_end
                        else:
                            # if is an independent entity and not inside a parenthesis
                            if parenthesis_count == 0:
                                sentence_entity_list.append(
                                    {"start": entity_start, "end": entity_end, "label": "Entity"})
                            else:
                                ignore_list.append(doc.text[entity_start:entity_end])
                        and_keyword_connecting = False

                    elif token.pos_ == "NOUN" and and_keyword_connecting:
                        # if is a noun (but not recognized as an entity) and should be connected by AND keyword
                        if sentence_entity_list and (token.idx + len(token) - sentence_entity_list[-1]['end'] <= 30):
                            sentence_entity_list[-1]['end'] = token.idx + len(token)
                        and_keyword_connecting = False

                    if token.text == 'and':
                        and_keyword_connecting = True

                    entity_start = -1
                    entity_end = -1

                # If we enter a parenthesis, all independent entities inside it should not be considered
                if token.text == "(":
                    parenthesis_count = parenthesis_count + 1
                elif token.text == ")":
                    parenthesis_count = parenthesis_count - 1

            for ent in sentence_entity_list:
                start_index = ent['start']
                end_index = ent['end']
                entity_text = doc.text[start_index:end_index]
                lower_text = entity_text.lower()

                # Only consider it to be an entity if it is not a simple word and has not occured within a parethesis
                should_add = True
                if lower_text in self.simple_dict or self.inflect_converter.singular_noun(
                        lower_text) in self.simple_dict:
                    should_add = False

                for ignore_word in ignore_list:
                    if ignore_word in entity_text:
                        should_add = False

                if should_add:
                    text_entity_list.append(ent)

        ex = {
            "text": doc.text,
            "ents": text_entity_list,
            "title": None
        }

        return ex

    def get_entity_pairs(self, ex):
        doc = self.nlp(ex['text'])
        period_idx_list = []
        # find the indices of the periods
        for token in doc:
            if token.text == ".":
                period_idx_list.append(token.idx)

        original_text = ex['text']
        entities = ex['ents']
        entity_pairs_with_confidence = []
        curr_sentence_entity_pairs = []

        for entity_index in range(0, len(entities) - 1):
            # Go over every ADJACENT entity pairs
            first_entity = entities[entity_index]
            second_entity = entities[entity_index + 1]
            if first_entity['start'] == second_entity['start']:
                continue

            period_in_between = 0
            for p in period_idx_list:
                if first_entity['start'] < p < second_entity['start']:
                    period_in_between += 1

            # Only consider entity pairs within the SAME sentence. If we reach a different sentence, compare all
            # entity pairs in the previous sentence and get the most confident pair
            if period_in_between > 0:
                if len(curr_sentence_entity_pairs) > 0:
                    curr_sentence_entity_pairs.sort(key=lambda x: x[3], reverse=True)  # Sort by confidence level
                    entity_pairs_with_confidence.append(
                        curr_sentence_entity_pairs[0])  # Only the most confident pair will remain
                    curr_sentence_entity_pairs = []
                continue

            # Find Previous Period and Next Period and use them to cut out sentence containing the two entities
            previous_period_order = 0  # Among all those periods, which one is right before the first entity
            while previous_period_order < len(period_idx_list) and period_idx_list[previous_period_order] < \
                    first_entity[
                        'start']:
                previous_period_order = previous_period_order + 1
            previous_period_order = previous_period_order - 1

            next_period_order = 0  # Among all those periods, which one is right after the second entity
            while next_period_order < len(period_idx_list) and period_idx_list[next_period_order] < second_entity[
                'end']:
                next_period_order = next_period_order + 1

            previous_period_pos = period_idx_list[previous_period_order]
            if previous_period_order == -1:
                previous_period_pos = -1

            if next_period_order == len(period_idx_list):
                next_period_pos = len(original_text) - 1
            else:
                next_period_pos = period_idx_list[next_period_order]

            relevant_sentence = original_text[previous_period_pos + 1: next_period_pos + 1]
            if relevant_sentence[0] == ' ':
                relevant_sentence = relevant_sentence[1:]
            # The indexes of the two entities in the cut out sentence
            first_entity_text = original_text[first_entity['start']:first_entity['end']]
            first_entity_start = relevant_sentence.index(first_entity_text)
            first_entity_end = first_entity_start + (first_entity['end'] - first_entity['start'])

            second_entity_text = original_text[second_entity['start']:second_entity['end']]
            second_entity_start = relevant_sentence.index(second_entity_text)
            second_entity_end = second_entity_start + (second_entity['end'] - second_entity['start'])

            # inferred relation format: ('father', 0.7654321)
            inferred_relation = self.opennre_model.infer({'text': relevant_sentence,
                                                          'h': {'pos': (first_entity_start, first_entity_end)},
                                                          't': {'pos': (second_entity_start, second_entity_end)}})

            confidence_level = inferred_relation[1]

            relevant_text_between = relevant_sentence[first_entity_end:second_entity_start]

            curr_sentence_entity_pairs.append((relevant_sentence, relevant_text_between,
                                               relevant_sentence[first_entity_start:first_entity_end],
                                               relevant_sentence[second_entity_start:second_entity_end],
                                               confidence_level))

        if len(curr_sentence_entity_pairs) > 0:
            curr_sentence_entity_pairs.sort(key=lambda x: x[4], reverse=True)  # Sort by confidence level
            entity_pairs_with_confidence.append(
                curr_sentence_entity_pairs[0])  # Only the most confident pair will remain
            curr_sentence_entity_pairs = []

        entity_pairs = []
        for entity_pair in entity_pairs_with_confidence:
            entity_pairs.append(
                {'sentence': entity_pair[0], 'text_between': entity_pair[1], 'first_entity': entity_pair[2],
                 'second_entity': entity_pair[3]})
        return entity_pairs

    @ProgressDecorator("Extracting Entities")
    def extract_entities(self, abstracts):
        total = 0
        entities_all_docs = []
        for i in tqdm(range(len(abstracts))):
            input_text = abstracts[i]
            ex = self.entity_extraction(input_text)
            entity_pairs = self.get_entity_pairs(ex)
            entities_all_docs.append(entity_pairs)
            total += len(entity_pairs)
        print(f"Number of abstracts processed: {len(abstracts)}")
        print(f"Number of entity pairs generated: {total}")
        return entities_all_docs
