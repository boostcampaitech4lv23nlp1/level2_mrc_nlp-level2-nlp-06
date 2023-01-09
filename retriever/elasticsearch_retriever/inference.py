import yaml
from elastic import ESRetriever

def main():
    with open("arg.yaml", "r") as f:
        arg = yaml.load(f, Loader=yaml.Loader)
    
    esretriever = ESRetriever()
    esretriever.connect()
    indices = esretriever.get_indices_name()
    if arg["index_name"] in indices:
        if arg["overwrite_index"]:
            esretriever.delete_index(arg["index_name"])
            esretriever.create_index(arg["index_name"],arg["similarity"],arg["wiki_path"])
            print("index deleted and new index created")
    else:
        esretriever.create_index(arg["index_name"],arg["similarity"],arg["wiki_path"])
        print("new index created")
    esretriever.get_relevant_doc(arg["index_name"],arg["topk"],arg["data_path"],arg["ouput_path"])


if __name__== "__main__":
    main()