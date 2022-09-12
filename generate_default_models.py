from federated_learning.arguments import Arguments
from federated_learning.nets import Cifar10CNN
from federated_learning.nets import FashionMNISTCNN
from federated_learning.nets import MNISTCNN
import os
import torch
from loguru import logger

if __name__ == '__main__':
    args = Arguments(logger)
    if not os.path.exists(args.get_default_model_folder_path()):
        os.mkdir(args.get_default_model_folder_path())

    # # ---------------------------------
    # # ----------- Cifar10CNN ----------
    # # ---------------------------------
    # full_save_path = os.path.join(args.get_default_model_folder_path(), "Cifar10CNN.model")
    # torch.save(Cifar10CNN().state_dict(), full_save_path)

    # # ---------------------------------
    # # -------- FashionMNISTCNN --------
    # # ---------------------------------
    # full_save_path = os.path.join(args.get_default_model_folder_path(), "FashionMNISTCNN.model")
    # torch.save(FashionMNISTCNN().state_dict(), full_save_path)

    # # ---------------------------------
    # # -------- MNISTCNN --------
    # # ---------------------------------
    # full_save_path = os.path.join(args.get_default_model_folder_path(), "MNISTCNN.model")
    # torch.save(MNISTCNN().state_dict(), full_save_path)

    # ---------------------------------
    # -------- QMNISTCNN --------
    # Note that we use the same model architecture for between QMNIST and MNIST
    # ---------------------------------
    full_save_path = os.path.join(args.get_default_model_folder_path(), "QMNISTCNN.model")
    torch.save(MNISTCNN().state_dict(), full_save_path)
