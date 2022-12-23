class Preprocess_features:
    def __init__(self, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

    def prepare_features(self, examples):
        tokenized_examples = self.tokenizer(
            examples["context"],
            truncation=True,
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        truncated = {"input_ids": [], "token_type_ids": [], "attention_mask": [], "start_index": [], "end_index": []}
        
        overflow_to_sample_mapping = tokenized_examples["overflow_to_sample_mapping"]
        offset_mapping = tokenized_examples["offset_mapping"]

        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        
        print("truncating datasets...")
        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]

            sequence_ids = tokenized_examples.sequence_ids(i)

            example_index = overflow_to_sample_mapping[i]
            answers = examples["answers"][example_index]

            answer_start_offset = answers["answer_start"][0]
            answer_end_offset = answer_start_offset + len(answers["text"][0])

            token_start_index = 0
            while sequence_ids[token_start_index] != 0:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 0:
                token_end_index -= 1

            if (
                offsets[token_start_index][0] <= answer_start_offset
                and offsets[token_end_index][1] >= answer_end_offset
            ):
                truncated["input_ids"].append(tokenized_examples["input_ids"][i])
                truncated["attention_mask"].append(tokenized_examples["attention_mask"][i])
                truncated["token_type_ids"].append(tokenized_examples["token_type_ids"][i])
                truncated["start_index"].append(token_start_index)
                truncated["end_index"].append(token_end_index)
        
        print("truncating completed!")
        
        return truncated
    