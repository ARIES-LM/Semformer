if __name__ == '__main__':
    import types
    from data import get_dataset
    from tokenizing import get_tokenizer

    # Create graphs and save
    n_train = 200000
    n_test = 20000
    deg = 2
    path_len = 20
    num_nodes = 50
    reverse = False
    generate_and_save(n_train=n_train, n_test=n_test, degSource=deg, pathLen=path_len, numNodes=num_nodes,
                      reverse=reverse)