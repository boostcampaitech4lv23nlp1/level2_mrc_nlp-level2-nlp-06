class Preprocess_features:
    def __init__(self, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride


    def process(self, train_dataset):
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

        overflow_to_sample_mapping = tokenized_contexts.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_contexts.pop("offset_mapping")

        tokenized_contexts["start_positions"] = []
        tokenized_contexts["end_positions"] = []
        tokenized_contexts["questions"] = []
        tokenized_contexts["answers"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_contexts["input_ids"][i]

            sequence_ids = tokenized_contexts.sequence_ids(i)

            example_index = overflow_to_sample_mapping[i]
            answers = train_dataset["answers"][example_index]

            answer_start_offset = answers["answer_start"][0]
            answer_end_offset = answer_start_offset + len(answers["text"][0])

            token_start_index = 0
            while sequence_ids[token_start_index] != 0:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 0:
                token_end_index -= 1

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
                tokenized_contexts["answers"].append(train_dataset["answers"][example_index])

        return tokenized_contexts
