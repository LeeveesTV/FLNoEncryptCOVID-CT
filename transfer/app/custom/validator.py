
import torch
import torchvision

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable

from torchvision.utils import make_grid
from torchvision import models as models 
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import ToTensor

from nvflare.apis.dxo import from_shareable, DataKind, DXO
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from simple_network import SimpleNetwork
import os

from PIL import Image
device = "cuda:0" if torch.cuda.is_available() else "cpu"

class Validator(Executor):
    
    def __init__(self, validate_task_name=AppConstants.TASK_VALIDATION):
        super(Validator, self).__init__()

        self._validate_task_name = validate_task_name

        # Setup the model
        self.model = SimpleNetwork()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)

        rootdir=os.getcwd() + "/data/"
        normalize = transforms.Normalize(mean=[0,0,0], std=[1,1,1])
        
        val_transformer = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize
        ])
        batchsize = 8
        testset = CovidCTDataset(root_dir = rootdir + '/Images-processed/',
                                  classes = ['CT_NonCOVID', 'CT_COVID'],
                                  covid_files=rootdir + '/Data-split/COVID/testCT_COVID.txt',
                                  non_covid_files=rootdir + '/Data-split/NonCOVID/testCT_NonCOVID.txt',
                                  transform= val_transformer)
        self.test_loader = DataLoader(testset, batch_size=batchsize, drop_last=False, shuffle=False)


    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if task_name == self._validate_task_name:
            model_owner = "?"
            try:
                try:
                    dxo = from_shareable(shareable)
                except:
                    self.log_error(fl_ctx, "Error in extracting dxo from shareable.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Ensure data_kind is weights.
                if not dxo.data_kind == DataKind.WEIGHTS:
                    self.log_exception(fl_ctx, f"DXO is of type {dxo.data_kind} but expected type WEIGHTS.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Extract weights and ensure they are tensor.
                model_owner = shareable.get_header(AppConstants.MODEL_OWNER, "?")
                weights = {k: torch.as_tensor(v, device=self.device) for k, v in dxo.data.items()}

                # Get validation accuracy
                val_accuracy = self.do_validation(weights, abort_signal)
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                self.log_info(fl_ctx, f"Accuracy when validating {model_owner}'s model on"
                                      f" {fl_ctx.get_identity_name()}"f's data: {val_accuracy}')

                dxo = DXO(data_kind=DataKind.METRICS, data={'val_acc': val_accuracy})
                return dxo.to_shareable()
            except:
                self.log_exception(fl_ctx, f"Exception in validating model from {model_owner}")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        else:
            return make_reply(ReturnCode.TASK_UNKNOWN)

    def do_validation(self, weights, abort_signal):
        self.model.load_state_dict(weights)

        self.model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                if abort_signal.triggered:
                    return 0

                images, labels = data['img'].to(device), data['label'].to(device)
                output = self.model(images)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.long().view_as(pred)).sum().item()

            metric = correct/(len(self.test_loader.dataset))
            print(metric)
        return metric

class CovidCTDataset():
    def __init__(self, root_dir, classes, covid_files, non_covid_files, transform=None):
        self.root_dir = root_dir
        self.classes = classes
        self.files_path = [non_covid_files, covid_files]
        self.image_list = []

        # read the files from data split text files
        covid_files = read_txt(covid_files)
        non_covid_files = read_txt(non_covid_files)

        # combine the positive and negative files into a cummulative files list
        for cls_index in range(len(self.classes)):
            
            class_files = [[os.path.join(self.root_dir, self.classes[cls_index], x), cls_index] \
                            for x in read_txt(self.files_path[cls_index])]
            self.image_list += class_files
                
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        path = self.image_list[idx][0]
        
        # Read the image
        image = Image.open(path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)

        label = int(self.image_list[idx][1])

        data = {'img':   image,
                'label': label,
                'paths' : path}

        return data

def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data