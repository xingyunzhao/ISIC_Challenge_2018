# Imports
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pathlib
from transformations import ComposeDouble, normalize_01, AlbuSeg2d, FunctionWrapperDouble, create_dense_target
from sklearn.model_selection import train_test_split
from customdatasets import SegmentationDataSet3
import torch
import numpy as np
from unet import UNet
from trainer import Trainer
from torch.utils.data import DataLoader
from skimage.transform import resize
import albumentations
from visual import plot_training

# root directory
root = pathlib.Path.cwd()
path_data = root / 'Data' / '2018'
path_output = root / "Output"
path_temp = root / "temp_chkp"

use_saved_data = True


def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext) if file.is_file()]
    return filenames


def main():
    if use_saved_data:
        print("Use saved datasets")
        dataset_train = torch.load(path_temp / 'dataset_train.pt')
        dataset_valid = torch.load(path_temp / 'dataset_valid.pt')

    else:
        print("Read and process new datasets")
        # input and target files
        inputs = get_filenames_of_path(path_data / 'ISIC2018_Task1-2_Training_Input', ext='*.jpg')
        targets = get_filenames_of_path(path_data / 'ISIC2018_Task1_Training_GroundTruth', ext='*.png')

        # use small set of data for demo
        # inputs = inputs[:200]
        # targets = targets[:200]

        # pre-transformations
        pre_transforms = ComposeDouble([
            FunctionWrapperDouble(resize,
                                  input=True,
                                  target=False,
                                  output_shape=(128, 128, 3)),
            FunctionWrapperDouble(resize,
                                  input=False,
                                  target=True,
                                  output_shape=(128, 128),
                                  order=0,
                                  anti_aliasing=False,
                                  preserve_range=True),
        ])

        # training transformations and augmentations
        transforms_training = ComposeDouble([
            AlbuSeg2d(albumentations.HorizontalFlip(p=0.5)),
            FunctionWrapperDouble(create_dense_target, input=False, target=True),
            FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0),
            FunctionWrapperDouble(normalize_01)
        ])

        # validation transformations
        transforms_validation = ComposeDouble([
            FunctionWrapperDouble(resize,
                                  input=True,
                                  target=False,
                                  output_shape=(128, 128, 3)),
            FunctionWrapperDouble(resize,
                                  input=False,
                                  target=True,
                                  output_shape=(128, 128),
                                  order=0,
                                  anti_aliasing=False,
                                  preserve_range=True),
            FunctionWrapperDouble(create_dense_target, input=False, target=True),
            FunctionWrapperDouble(np.moveaxis, input=True, target=False, source=-1, destination=0),
            FunctionWrapperDouble(normalize_01)
        ])

        # random seed
        random_seed = 42

        # split dataset into training set and validation set
        train_size = 0.9

        inputs_train, inputs_valid, targets_train, targets_valid = train_test_split(
            inputs,
            targets,
            random_state=random_seed,
            train_size=train_size,
            shuffle=True)

        # dataset training
        dataset_train = SegmentationDataSet3(inputs=inputs_train,
                                            targets=targets_train,
                                            transform=transforms_training,
                                            use_cache=True,
                                            pre_transform=pre_transforms)

        # dataset validation
        dataset_valid = SegmentationDataSet3(inputs=inputs_valid,
                                            targets=targets_valid,
                                            transform=transforms_validation,
                                            use_cache=True,
                                            pre_transform=pre_transforms)

        # save dataset
        torch.save(dataset_train, path_temp / 'dataset_train.pt')
        torch.save(dataset_valid, path_temp / 'dataset_valid.pt')


    # dataloader training
    dataloader_training = DataLoader(dataset=dataset_train,
                                     batch_size=20,
                                     shuffle=True)

    # dataloader validation
    dataloader_validation = DataLoader(dataset=dataset_valid,
                                       batch_size=20,
                                       shuffle=True)


    # device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # model
    model = UNet(in_channels=3,
                 out_channels=2,
                 n_blocks=4,
                 start_filters=32,
                 activation='relu',
                 normalization='batch',
                 conv_mode='same',
                 dim=2).to(device)

    # criterion
    criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    optimizer = torch.optim.Adam(model.parameters())

    # trainer
    trainer = Trainer(model=model,
                      device=device,
                      criterion=criterion,
                      optimizer=optimizer,
                      training_DataLoader=dataloader_training,
                      validation_DataLoader=dataloader_validation,
                      lr_scheduler=None,
                      epochs=50,
                      epoch=0,
                      notebook=False)

    # start training
    print("Start training")
    training_losses, validation_losses, lr_rates = trainer.run_trainer()

    # save the model
    model_name = 'model_ep-50_Adam.pt'
    torch.save(model.state_dict(), path_output / model_name)

    # Plot results
    fig = plot_training(training_losses, validation_losses, lr_rates, gaussian=True, sigma=1, figsize=(10, 4))
    fig.savefig(path_output / "model_ep-50_Adam.jpg")


    # # Learning rate finder
    # # device
    # if torch.cuda.is_available():
    #     device = torch.device('cuda')
    # else:
    #     torch.device('cpu')
    #
    # # model
    # model = UNet(in_channels=3,
    #              out_channels=2,
    #              n_blocks=4,
    #              start_filters=32,
    #              activation='relu',
    #              normalization='batch',
    #              conv_mode='same',
    #              dim=2).to(device)
    #
    # # criterion
    # criterion = torch.nn.CrossEntropyLoss()
    #
    # # optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    #
    #
    # from lr_rate_finder import LearningRateFinder
    # lrf = LearningRateFinder(model, criterion, optimizer, device)
    # lrf.fit(dataloader_training, steps=1000)
    # lrf.plot()


if __name__ == '__main__':
    main()


