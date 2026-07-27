import os
import cv2 as cv

import torch
from torch.utils.data import Dataset



class AdinkraDataset(Dataset):


    def __init__(self,dataset_folder):

        self.images=[]
        self.labels=[]


        symbols = sorted(
            os.listdir(dataset_folder)
        )


        self.label_map={}


        for i,symbol in enumerate(symbols):

            self.label_map[symbol]=i



        for symbol in symbols:


            folder=os.path.join(
                dataset_folder,
                symbol
            )


            for img in os.listdir(folder):

                self.images.append(
                    os.path.join(folder,img)
                )


                self.labels.append(
                    self.label_map[symbol]
                )



    def __len__(self):

        return len(self.images)



    def __getitem__(self,index):

        path=self.images[index]

        label=self.labels[index]


        img=cv.imread(path)


        img=cv.cvtColor(
            img,
            cv.COLOR_BGR2RGB
        )


        img=img/255.0


        img=torch.tensor(
            img,
            dtype=torch.float32
        )


        img=img.permute(
            2,0,1
        )


        return img,label