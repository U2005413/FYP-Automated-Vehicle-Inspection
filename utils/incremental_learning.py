from datetime import datetime


def incremental_learning(model, dataset_yaml_path, training_epochs):
    now = datetime.now()
    current_time = now.strftime("%d_%b_%Y %H%M")

    if training_epochs % 2 != 0:
        training_epochs += 1
    epochs = int(training_epochs / 2)

    model.train(
        data=dataset_yaml_path,
        epochs=epochs,
        freeze=10,
        save=False,
        val=False,
        plots=False,
    )
    model.train(
        data=dataset_yaml_path,
        epochs=epochs,
        freeze=0,
        save=True,
        val=True,
        plots=True,
        project="trains",
        name=current_time,
    )

    return "trains\\" + current_time
