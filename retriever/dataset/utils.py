class Preprocess_features:
    def __init__(self, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

    def process(self, train_dataset):
        offset_newline = [context[:train_dataset["answers"][i]["answer_start"][0]].count("\\n") for i, context in enumerate(train_dataset["context"])]
        contexts = [context.replace("\\n", "") for context in train_dataset["context"]]
        
        tokenized_contexts = self.tokenizer(
            contexts,
            truncation=True,
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = tokenized_contexts.pop("offset_mapping")

        tokenized_contexts["start_positions"] = []
        tokenized_contexts["end_positions"] = []
        tokenized_contexts["questions"] = []
        tokenized_contexts["answers"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_contexts["input_ids"][i]

            sequence_ids = tokenized_contexts.sequence_ids(i)

            example_index = tokenized_contexts["overflow_to_sample_mapping"][i]
            answers = train_dataset["answers"][example_index]

            answer_start_offset = answers["answer_start"][0] - offset_newline[example_index] * 2
            answer_end_offset = answer_start_offset + len(answers["text"][0])

            token_start_index = 0
            while sequence_ids[token_start_index] != 0:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 0:
                token_end_index -= 1
            
            subdocument_start_index = offsets[token_start_index][0]

            if not (
                offsets[token_start_index][0] <= answer_start_offset
                and offsets[token_end_index][1] >= answer_end_offset
            ):
                tokenized_contexts["start_positions"].append(-1)
                tokenized_contexts["end_positions"].append(-1)
                tokenized_contexts["questions"].append(None)
                tokenized_contexts["answers"].append(None)
            else:
                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= answer_start_offset
                ):
                    token_start_index += 1
                while offsets[token_end_index][1] >= answer_end_offset:
                    token_end_index -= 1
                tokenized_contexts["start_positions"].append(token_start_index - 1)
                tokenized_contexts["end_positions"].append(token_end_index + 1)
                tokenized_contexts["questions"].append(train_dataset["question"][example_index])
                tokenized_contexts["answers"].append({
                    "text": train_dataset["answers"][example_index]["text"],
                    "answer_start": [answer_start_offset - subdocument_start_index]
                })

        return tokenized_contexts

    def get_hard_negatives(self, dataset, tokenized_passages, hard_negative_nums, hn_df):
        '''
        dataset (datasets): wiki train dataset
        tokenized_passages (datasets): result of self.preprocess()
        hard_negative_nums (int): number of hard negatives.
        hn_df (pd.DataFrame): DataFrame of hard negative csv file.
        '''
        hard_negatives = []

        for i, question in enumerate(dataset['question']):
            hard_negative = list(hn_df[hn_df['hard_negative']==question])
            hard_negative = hard_negative[:hard_negative_nums]
            hard_negatives += hard_negative
        hard_negatives = [n.replace("\\n", "").replace("\n", "") for n in hard_negatives]

        tokenized_hard_negatives = self.tokenizer(
            hard_negatives,
            truncation=True,
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        tokenized_hn = {'input_ids': [], 'token_type_ids': [], "attention_mask": []}

        for i in tokenized_passages['overflow_to_sample_mapping']:
            hn_index = tokenized_hard_negatives['overflow_to_sample_mapping'].index(i)
            tokenized_hn['input_ids'] += tokenized_hard_negatives['input_ids'][hn_index:hn_index+hard_negative_nums]
            tokenized_hn['token_type_ids'] += tokenized_hard_negatives['token_type_ids'][hn_index:hn_index+hard_negative_nums]
            tokenized_hn['attention_mask'] += tokenized_hard_negatives['attention_mask'][hn_index:hn_index+hard_negative_nums]
            
        return tokenized_hn
    