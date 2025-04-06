from datasets import Dataset, DatasetDict

def create_train_datasets(df):

    def preprocess(samples):

        batch = []
        PROMPT_DICT = {

        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context.\n"
            "아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n"
            "Write a response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 작성하세요.\n\n"
            "### Instruction(명령어):{instruction}\n\n### Input(입력):{input}\n\n### Response(응답):{response}<eos>"
        )
    }

        for instruction, input, output in zip(samples["instruction"], samples["input"], samples["output"]):
            user_input = input
            response = output
            conversation = PROMPT_DICT['prompt_input'].replace('{instruction}', instruction[0]).replace('{input}', user_input).replace('{response}', response)
            batch.append(conversation)

        return {"text": batch}

    def generate_dict(df) :

        instruction_list = [ [open('../data/instruction.txt').read()] for _ in range(len(df)) ]
        input_list = df['restore_review']
        output_list = df['output']
        
        dataset_dict = {'instruction' : instruction_list , 'input' : input_list , 'output' : output_list}
        dataset = Dataset.from_dict(dataset_dict)

        return dataset

    dataset = generate_dict(df)
    raw_datasets = DatasetDict()
    datasets = dataset.train_test_split(test_size = 0.005,
                                        shuffle= True ,
                                        seed = 42)

    raw_datasets['train'] = datasets['train']
    raw_datasets['test'] = datasets['test']

    raw_datasets = raw_datasets.map(
        preprocess,
        batched = True,
        remove_columns=raw_datasets['train'].column_names
    )

    train_data = raw_datasets["train"]
    valid_data = raw_datasets["test"]

    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
    print(f"A sample of train dataset: {train_data[0]}")

    return train_data, valid_data

def create_test_datasets(df):

    def preprocess(samples):

        batch = []
        PROMPT_DICT = {

        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context.\n"
            "아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n"
            "Write a response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 작성하세요.\n\n"
            "### Instruction(명령어):{instruction}\n\n### Input(입력):{input}\n\n### Response(응답):"
        )
    }

        for instruction, input in zip(samples["instruction"], samples["input"]):
            user_input = input
            conversation = PROMPT_DICT['prompt_input'].replace('{instruction}', instruction[0]).replace('{input}', user_input)
            batch.append(conversation)

        return {"text": batch}

    def generate_dict(df) :

        instruction_list = [ [open('../data/instruction.txt').read()] for _ in range(len(df)) ]
        input_list = df['input']
        dataset_dict = {'instruction' : instruction_list , 'input' : input_list }
        dataset = Dataset.from_dict(dataset_dict)

        return dataset

    datasets = generate_dict(df)
    raw_datasets = DatasetDict()

    raw_datasets['test'] = datasets

    raw_datasets = raw_datasets.map(
        preprocess,
        batched = True,
        remove_columns=['instruction']
    )

    test_data = raw_datasets["test"]

    print(f"Size of the test set: {len(test_data)}")
    print(f"A sample of test dataset: {test_data[0]}")

    return test_data