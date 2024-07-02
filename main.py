import trainer

if __name__ == '__main__':

    training = False

    trainer.train_model(train=training)
    # batch_sizes = [2, 4, 8]
    # for batch_size in batch_sizes:
    #     trainer.train_model(batch_size=batch_size, model_name=('batch_size_' + str(batch_size)))