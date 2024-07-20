import trainer

if __name__ == '__main__':

    seeds = [5, 42, 101]

    for seed in seeds:
        trainer.train_model(nb_epochs=10, load_checkpoint=False, 
                            model_name=f'19_Jul_2024_no_spec_arg_test_seed_{seed}', random_seed=seed,
                            train=True, spec_arg=False)
        trainer.train_model(nb_epochs=10, load_checkpoint=True, 
                            model_name=f'19_Jul_2024_spec_arg_test_seed_{seed}', random_seed=seed,
                            train=True, spec_arg=True)

    # batch_sizes = [2, 4, 8]
    # for batch_size in batch_sizes:
    #     trainer.train_model(batch_size=batch_size, model_name=('batch_size_' + str(batch_size)))