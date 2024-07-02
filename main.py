import trainer

if __name__ == '__main__':

    trainer.train_model(nb_epochs=5, load_checkpoint=True, model_name='cm_test', train=False)
    # batch_sizes = [2, 4, 8]
    # for batch_size in batch_sizes:
    #     trainer.train_model(batch_size=batch_size, model_name=('batch_size_' + str(batch_size)))