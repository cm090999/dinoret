# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import os
import random
import json
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings

from util import preprocessing

def build_transform(is_train, args):
    mean = args.normalize_mean
    std = args.normalize_std

    ## Preprocessing Transforms
    t = []

    # Numpy operation
    if (args.preprocessing_CropRoundImage == True) or (args.preprocessing_CLAHETransform == True) or (args.preprocessing_MedianFilterTransform == True):
        t.append(preprocessing.PILToNumpy())

    # Crop Black borders
    if args.preprocessing_CropRoundImage == True:
        t.append(preprocessing.CropRoundImage())

    # Equalize Images
    if args.preprocessing_CLAHETransform == True:
        t.append(preprocessing.CLAHETransform())

    # Apply Median Filter 
    if args.preprocessing_MedianFilterTransform == True:
        t.append(preprocessing.MedianFilterTransform())

    # End Numpy operations
    if (args.preprocessing_CropRoundImage == True) or (args.preprocessing_CLAHETransform == True) or (args.preprocessing_MedianFilterTransform == True):
        t.append(preprocessing.NumpyToPIL())

    t.append(
        transforms.Resize((args.input_size, args.input_size), interpolation=transforms.InterpolationMode.BICUBIC), 
    )

    # Training Augmentations
    if is_train=='train':
        t.append(transforms.RandomHorizontalFlip(p = args.Vertical_Horizontal_Flip_Probability))
        t.append(transforms.RandomVerticalFlip(p = args.Vertical_Horizontal_Flip_Probability))
        t.append(transforms.RandomRotation(degrees=args.RandomRotation_degrees))
        t.append(transforms.RandomAdjustSharpness(sharpness_factor=args.RandomAdjustSharpness_sharpness_factor))
        t.append(transforms.ColorJitter(brightness=args.color_jitter_param_brightness, contrast=args.color_jitter_param_contrast, saturation=args.color_jitter_param_saturation, hue=args.color_jitter_param_hue))
        t.append(transforms.RandomResizedCrop(size=(args.input_size, args.input_size), scale=[args.RandomResizeCrop_lowerBound, args.RandomResizeCrop_upperBound]))

    # Basic Transforms
    t.append(transforms.ToTensor())

    t.append(transforms.Normalize(mean, std))    

    return transforms.Compose(t)

def get_datasets(args):

    if 'aptos' in args.data_path.lower():
        train_df, val_df, test_df, dataset_constructor = APTOS2019_Dataset.get_dataframe_split(args)
    elif 'eyepacs' in args.data_path.lower():
        train_df, val_df, test_df, dataset_constructor = EYEPACS_Dataset.get_dataframe_split(args)
    elif 'idrid' in args.data_path.lower():
        train_df, val_df, test_df, dataset_constructor = IDRiD_Dataset.get_dataframe_split(args)
    elif 'messidor' in args.data_path.lower():
        train_df, val_df, test_df, dataset_constructor = Messidor_Dataset.get_dataframe_split(args)
    elif 'drtid' in args.data_path.lower():
        train_df, val_df, test_df, dataset_constructor = DRTiD_Dataset.get_dataframe_split(args)
    elif 'papila' in args.data_path.lower():
        train_df, val_df, test_df, dataset_constructor = Papila_Dataset.get_dataframe_split(args)
    else:
        raise NotImplementedError()

    # Split into train and aptosvalidation
    if args.few_shot_learning >= 1:
        train_df = create_few_shot_learning_dataframe(args, train_df)

    # Create datasets
    train_dataset, val_dataset, test_dataset = dataset_constructor(args, train_df, val_df, test_df)

    if args.few_shot_learning != -1:
        # Save the paths of the chosen samples TODO: Check for cross-validation
        samples_paths = [train_dataset.image_ids, train_dataset.targets]
        with open(os.path.join(args.task, 'few_shot_learning_samples.txt'), 'w') as txt_file:
            json.dump(samples_paths, txt_file, indent=4)

    print("Train dataset size: %d" % len(train_dataset))
    print("Validation dataset size: %d" % len(val_dataset))
    print("Test dataset size: %d" % len(test_dataset))

    return train_dataset, val_dataset, test_dataset

def create_few_shot_learning_dataframe(args, train_df):

    # Split into train and validation according to args.few_shot_learning
    minority_class_size = train_df['class_label'].value_counts().min()

    # Abort if args.few_shot_learning > minority_class_size * 0.85
    if args.few_shot_learning > minority_class_size:
        warnings.warn("args.few_shot_learning > minority_class_size. Aborting.")
        exit()

    # Get indices of each class
    class_indices = [[index for index, element in enumerate(train_df['class_label']) if element == i] for i in range(args.nb_classes)]

    # Shuffle indices of each class
    local_random = random.Random(args.few_shot_learning_seed)
    class_indices_shuffled = [sorted(class_indices[i], key=lambda x: local_random.random()) for i in range(len(class_indices))]

    # Get first args.few_shot_learning indices of each class
    chosen_idcs_ = [class_indices_shuffled[i][:args.few_shot_learning] for i in range(len(class_indices_shuffled))]
    chosen_idcs = [chosen_idcs_[i][j] for i in range(len(chosen_idcs_)) for j in range(len(chosen_idcs_[i]))]
    
    # Get train and validation dataframes
    train_df = train_df.iloc[chosen_idcs]

    # Reset indices
    train_df.reset_index(drop=True, inplace=True)

    return train_df

class BaseDataset(Dataset):
    def __init__(self):
        pass
        return
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):

        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get label
        label = self.targets[idx]

        # Get image id
        image_path_rel = self.image_paths_rel[idx]

        return image, label, image_path_rel, image_path

class APTOS2019_Dataset(BaseDataset):
    def __init__(self, dataset_dir, dataframe, transform=None):
        super().__init__()
        
        self.classes = ['0_No_DR', '1_Mild_DR', '2_Moderate_DR', '3_Severe_DR', '4_Proliferative_DR']

        self.dataset_dir = dataset_dir
        self.dataframe = dataframe
        self.transform = transform

        # Get image ids and labels
        self.image_ids = self.dataframe['image_name'].values.tolist()
        self.targets = self.dataframe['class_label'].values.tolist()

        # Create image paths
        self.image_paths = [os.path.join(self.dataset_dir,'train_images' ,image_id + '.png') for image_id in self.image_ids]

        self.image_paths_rel = [os.path.join('train_images' , image_id + '.png') for image_id in self.image_ids]
        return
    
    
    @staticmethod
    def get_dataframe_split(args):
        

        if args.use_filtered_data:
            # Remove rows with ungradable images
            aptos_df = pd.read_csv(os.path.join(args.data_path, 'train_filtered.csv'))
            aptos_df = aptos_df[aptos_df['Quality'] != 0]
        else:
            aptos_df = pd.read_csv(os.path.join(args.data_path, 'train.csv'))
        

        # Rename columns
        aptos_df.rename(columns={'id_code':'image_name', 'diagnosis':'class_label'}, inplace=True)

        # Split into train and Test
        fraction = 0.15
        train_val_df, test_df = train_test_split(aptos_df, test_size=fraction, random_state=99, stratify=aptos_df['class_label'])
        corrected_fraction = fraction / (1.0 - fraction) #len(train_val_df) / len(aptos_df)
        train_df, val_df = train_test_split(train_val_df, test_size=corrected_fraction, random_state=99, stratify=train_val_df['class_label'])
    
        # Reset indices
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)
        
        def class_constructor(args, train_df, val_df, test_df):
            # Create datasets
            train_dataset = APTOS2019_Dataset(args.data_path, train_df, transform=build_transform('train', args))
            val_dataset = APTOS2019_Dataset(args.data_path, val_df, transform=build_transform('val', args))
            test_dataset = APTOS2019_Dataset(args.data_path, test_df, transform=build_transform('test', args))

            return train_dataset, val_dataset, test_dataset

        return train_df, val_df, test_df, class_constructor


class EYEPACS_Dataset(BaseDataset):
    def __init__(self, dataset_dir, dataframe, transform=None):
        super().__init__()
        
        self.classes = ['0_No_DR', '1_Mild_DR', '2_Moderate_DR', '3_Severe_DR', '4_Proliferative_DR']

        self.dataset_dir = dataset_dir
        self.dataframe = dataframe
        self.transform = transform

        # Get image ids and labels
        self.image_ids = self.dataframe['image_name'].values.tolist()
        self.targets = self.dataframe['class_label'].values.tolist()

        self.image_type_extension = ".jpeg"
        self.image_folder_name = self._get_correct_folder_path()

        # Create image paths
        self.image_paths = [os.path.join(self.dataset_dir, self.image_folder_name , image_id + self.image_type_extension) for image_id in self.image_ids]

        self.image_paths_rel = [os.path.join(self.image_folder_name , image_id + self.image_type_extension) for image_id in self.image_ids]
        return
    
    def _get_correct_folder_path(self):
        # check if first image exists in one folder
        if os.path.exists(os.path.join(self.dataset_dir,'train' ,self.image_ids[0] + self.image_type_extension)):
            return "train"
        elif os.path.exists(os.path.join(self.dataset_dir,'test' ,self.image_ids[0] + self.image_type_extension)):
            return "test"
        else:
            raise FileNotFoundError('Could not find image folder')
        
    @staticmethod
    def get_dataframe_split(args):
        train_val_df = pd.read_csv(os.path.join(args.data_path, 'trainLabels.csv'))
        test_df = pd.read_csv(os.path.join(args.data_path, 'testLabels.csv'))

        train_val_df.rename(columns={'image':'image_name', 'level':'class_label'}, inplace=True)
        test_df.rename(columns={'image':'image_name', 'level':'class_label'}, inplace=True)


        val_fraction = 0.10
        train_df, val_df = train_test_split(train_val_df, test_size=val_fraction, random_state=99, stratify=train_val_df['class_label'])

        # Reset indices
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)

        def class_constructor(args, train_df, val_df, test_df):
            # Create datasets
            train_dataset = EYEPACS_Dataset(args.data_path, train_df, transform=build_transform('train', args))
            val_dataset = EYEPACS_Dataset(args.data_path, val_df, transform=build_transform('val', args))
            test_dataset = EYEPACS_Dataset(args.data_path, test_df, transform=build_transform('test', args))
            return train_dataset, val_dataset, test_dataset
        
        return train_df, val_df, test_df, class_constructor
    

class IDRiD_Dataset(BaseDataset):
    def __init__(self, dataset_dir, dataframe, is_test, transform=None):
        super().__init__()

        self.classes = ['0_No_DR', '1_Mild_DR', '2_Moderate_DR', '3_Severe_DR', '4_Proliferative_DR']

        if is_test == True:
            self.mode = 'test'
        else:
            self.mode = 'train'

        self.dataset_dir = dataset_dir
        self.dataframe = dataframe
        self.transform = transform

        # Get image ids and labels
        self.image_ids = self.dataframe['image_name'].values.tolist()
        self.targets = self.dataframe['class_label'].values.tolist()

        self.image_type_extension = ".jpg"
        if self.mode == 'train':
            self.image_folder_name = "1. Original Images/a. Training Set"
        elif self.mode == 'test':
            self.image_folder_name = "1. Original Images/b. Testing Set"

        # Create image paths
        self.image_paths = [os.path.join(self.dataset_dir, self.image_folder_name , image_id + self.image_type_extension) for image_id in self.image_ids]

        self.image_paths_rel = [os.path.join(self.image_folder_name , image_id + self.image_type_extension) for image_id in self.image_ids]

        return
    
    @staticmethod
    def get_dataframe_split(args):

        if args.use_filtered_data:
            # Remove rows with ungradable images
            train_val_df = pd.read_csv(os.path.join(args.data_path, '2. Groundtruths/a. IDRiD_Disease Grading_Training Labels_filtered.csv'))
            test_df = pd.read_csv(os.path.join(args.data_path, '2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels_filtered.csv'))
            train_val_df = train_val_df[train_val_df['Quality'] != 0]
            test_df = test_df[test_df['Quality'] != 0]
        else:
            train_val_df = pd.read_csv(os.path.join(args.data_path, '2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv'))
            test_df = pd.read_csv(os.path.join(args.data_path, '2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv'))

        train_val_df.rename(columns={'Image name':'image_name', 'Retinopathy grade':'class_label'}, inplace=True)
        test_df.rename(columns={'Image name':'image_name', 'Retinopathy grade':'class_label'}, inplace=True)


        val_fraction = 0.15 #same as test
        corrected_fraction = val_fraction / (1.0 - val_fraction)
        train_df, val_df = train_test_split(train_val_df, test_size=corrected_fraction, random_state=99, stratify=train_val_df['class_label'])

        # Reset indices
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)

        def class_constructor(args, train_df, val_df, test_df):
            # Create datasets
            train_dataset = IDRiD_Dataset(args.data_path, train_df, is_test = False, transform=build_transform('train', args))
            val_dataset = IDRiD_Dataset(args.data_path, val_df, is_test = False, transform=build_transform('val', args))
            test_dataset = IDRiD_Dataset(args.data_path, test_df, is_test = True, transform=build_transform('test', args))
            return train_dataset, val_dataset, test_dataset
        
        return train_df, val_df, test_df, class_constructor

    
class Messidor_Dataset(BaseDataset):
    def __init__(self, dataset_dir, dataframe, transform=None):
        super().__init__()
        
        self.classes = ['0_No_DR', '1_Mild_DR', '2_Moderate_DR', '3_Severe_DR', '4_Proliferative_DR']

        self.dataset_dir = dataset_dir
        self.dataframe = dataframe
        self.transform = transform
        self.image_type_extension = ""

        # Get image ids and labels
        self.image_ids = self.dataframe['image_name'].values.tolist()
        self.targets = self.dataframe['class_label'].values.tolist()

        # Create image paths
        self.image_paths = [os.path.join(self.dataset_dir,'images' ,image_id + self.image_type_extension) for image_id in self.image_ids]

        self.image_paths_rel = [os.path.join('images' , image_id + self.image_type_extension) for image_id in self.image_ids]
        return
    
    
    @staticmethod
    def get_dataframe_split(args):

        if args.use_filtered_data:
            # Remove rows with ungradable images
            messidor_df = pd.read_csv(os.path.join(args.data_path, 'messidor_data_filtered.csv'))
            messidor_df = messidor_df[messidor_df['Quality'] != 0]
        else:
            messidor_df = pd.read_csv(os.path.join(args.data_path, 'messidor_data.csv'))
        
        # Rename columns
        messidor_df.rename(columns={'image_id':'image_name', 'adjudicated_dr_grade':'class_label'}, inplace=True)

        # Remove rows with ungradable images
        messidor_df = messidor_df[messidor_df['adjudicated_gradable'] == 1]

        # Print samples with NaN or None class labels
        print("Samples with NaN or None class labels:")
        print(messidor_df[messidor_df['class_label'].isnull()])

        # Ensure type of target is int
        messidor_df['class_label'] = messidor_df['class_label'].astype(int)

        # Split into train and Test
        fraction = 0.15 
        train_val_df, test_df = train_test_split(messidor_df, test_size=fraction, random_state=99, stratify=messidor_df['class_label'])
        corrected_fraction = fraction / (1.0 - fraction) #len(train_val_df) / len(aptos_df)
        train_df, val_df = train_test_split(train_val_df, test_size=corrected_fraction, random_state=99, stratify=train_val_df['class_label'])
    
        # Reset indices
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)
        
        def class_constructor(args, train_df, val_df, test_df):
            # Create datasets
            train_dataset = Messidor_Dataset(args.data_path, train_df, transform=build_transform('train', args))
            val_dataset = Messidor_Dataset(args.data_path, val_df, transform=build_transform('val', args))
            test_dataset = Messidor_Dataset(args.data_path, test_df, transform=build_transform('test', args))

            return train_dataset, val_dataset, test_dataset

        return train_df, val_df, test_df, class_constructor
    
class DRTiD_Dataset(BaseDataset):
    def __init__(self, dataset_dir, dataframe, is_test, transform=None):
        super().__init__()

        self.classes = ['0_No_DR', '1_Mild_DR', '2_Moderate_DR', '3_Severe_DR', '4_Proliferative_DR']

        if is_test == True:
            self.mode = 'test'
        else:
            self.mode = 'train'

        self.dataset_dir = dataset_dir
        self.dataframe = dataframe
        self.transform = transform

        # Get image ids and labels
        self.image_ids = self.dataframe['image_name'].values.tolist()
        self.targets = self.dataframe['class_label'].values.tolist()

        self.image_type_extension = ".jpg"

        self.image_folder_name = "Original Images"

        # Create image paths
        self.image_paths = [os.path.join(self.dataset_dir, self.image_folder_name, image_id + self.image_type_extension) for image_id in self.image_ids]

        self.image_paths_rel = [os.path.join(self.image_folder_name, image_id + self.image_type_extension) for image_id in self.image_ids]

        return
    
    @staticmethod
    def get_dataframe_split(args):

        if args.use_filtered_data:
            # Remove rows with ungradable images
            train_val_df = pd.read_csv(os.path.join(args.data_path, 'Ground Truths/DR_grade/a. DR_grade_Training_filtered.csv'))
            test_df = pd.read_csv(os.path.join(args.data_path, 'Ground Truths/DR_grade/b. DR_grade_Testing_filtered.csv'))
            train_val_df = train_val_df[train_val_df['Quality'] != 0]
            test_df = test_df[test_df['Quality'] != 0]
        else:
            train_val_df = pd.read_csv(os.path.join(args.data_path, 'Ground Truths/DR_grade/a. DR_grade_Training.csv'))
            test_df = pd.read_csv(os.path.join(args.data_path, 'Ground Truths/DR_grade/b. DR_grade_Testing.csv'))

        train_val_df.rename(columns={'Macula':'image_name', 'Grade':'class_label'}, inplace=True)
        test_df.rename(columns={'Macula':'image_name', 'Grade':'class_label'}, inplace=True)


        val_fraction = 0.15 #same as test
        corrected_fraction = val_fraction / (1.0 - val_fraction)
        train_df, val_df = train_test_split(train_val_df, test_size=corrected_fraction, random_state=99, stratify=train_val_df['class_label'])

        # Reset indices
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)

        def class_constructor(args, train_df, val_df, test_df):
            # Create datasets
            train_dataset = DRTiD_Dataset(args.data_path, train_df, is_test = False, transform=build_transform('train', args))
            val_dataset = DRTiD_Dataset(args.data_path, val_df, is_test = False, transform=build_transform('val', args))
            test_dataset = DRTiD_Dataset(args.data_path, test_df, is_test = True, transform=build_transform('test', args))
            return train_dataset, val_dataset, test_dataset
        
        return train_df, val_df, test_df, class_constructor


class Papila_Dataset(BaseDataset):
    def __init__(self, dataset_dir, dataframe, transform=None):
        super().__init__()
        
        self.classes = ['0_non_glaucomatous', '1_suspect', '2_glaucomatous']

        self.dataset_dir = dataset_dir
        self.dataframe = dataframe
        self.transform = transform
        self.image_type_extension = ".jpg"

        # Get image ids and labels
        self.image_ids = self.dataframe['image_name'].values.tolist()
        self.targets = self.dataframe['class_label'].values.tolist()

        # Create image paths
        self.image_paths = [os.path.join(self.dataset_dir,'FundusImages' ,image_id + self.image_type_extension) for image_id in self.image_ids]

        self.image_paths_rel = [os.path.join('FundusImages' , image_id + self.image_type_extension) for image_id in self.image_ids]
        return
    
    
    @staticmethod
    def get_dataframe_split(args):

        if args.use_filtered_data:
            # Remove rows with ungradable images
            df_od = pd.read_csv(os.path.join(args.data_path, 'ClinicalData/patient_data_od_filtered.csv'))
            df_os = pd.read_csv(os.path.join(args.data_path, 'ClinicalData/patient_data_os_filtered.csv'))
            df_od = df_od[df_od['Quality'] != 0]
            df_os = df_os[df_os['Quality'] != 0]
        else:
            df_od = pd.read_csv(os.path.join(args.data_path, 'ClinicalData/patient_data_od.csv'))
            df_os = pd.read_csv(os.path.join(args.data_path, 'ClinicalData/patient_data_os.csv'))
        
        # Rename columns
        df_od.rename(columns={'ID':'image_name', 'Diagnosis':'class_label'}, inplace=True)
        df_os.rename(columns={'ID':'image_name', 'Diagnosis':'class_label'}, inplace=True)

        # Convert ID column to string
        df_od['image_name'] = df_od['image_name'].astype(str)
        df_os['image_name'] = df_os['image_name'].astype(str)

        # Remove # from image_name
        df_od['image_name'] = df_od['image_name'].str.replace('#', '')
        df_os['image_name'] = df_os['image_name'].str.replace('#', '')

        # Add OS and OD to image_name
        df_od['image_name'] = 'RET' + df_od['image_name'] + 'OD'
        df_os['image_name'] = 'RET' + df_os['image_name'] + 'OS'

        # Split left train and Test
        fraction = 0.15
        train_val_df_os, test_df_os = train_test_split(df_os, test_size=fraction, random_state=99, stratify=df_os['class_label'])
        corrected_fraction = fraction / (1.0 - fraction) #len(train_val_df) / len(aptos_df)
        train_df_os, val_df_os = train_test_split(train_val_df_os, test_size=corrected_fraction, random_state=99, stratify=train_val_df_os['class_label'])

        # Apply same split to right
        train_df_od = df_od.iloc[train_df_os.index]
        val_df_od = df_od.iloc[val_df_os.index]
        test_df_od = df_od.iloc[test_df_os.index]

        # Reset indices
        train_df_os.reset_index(drop=True, inplace=True)
        test_df_os.reset_index(drop=True, inplace=True)
        val_df_os.reset_index(drop=True, inplace=True)
        train_df_od.reset_index(drop=True, inplace=True)
        test_df_od.reset_index(drop=True, inplace=True)
        val_df_od.reset_index(drop=True, inplace=True)

        # Concatenate left and right
        train_df = pd.concat([train_df_os, train_df_od])
        val_df = pd.concat([val_df_os, val_df_od])
        test_df = pd.concat([test_df_os, test_df_od])

        
        def class_constructor(args, train_df, val_df, test_df):
            # Create datasets
            train_dataset = Papila_Dataset(args.data_path, train_df, transform=build_transform('train', args))
            val_dataset = Papila_Dataset(args.data_path, val_df, transform=build_transform('val', args))
            test_dataset = Papila_Dataset(args.data_path, test_df, transform=build_transform('test', args))

            return train_dataset, val_dataset, test_dataset

        return train_df, val_df, test_df, class_constructor
