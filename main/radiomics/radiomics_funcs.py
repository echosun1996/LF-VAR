import math
import os
import shutil

import PIL
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import SimpleITK as sitk
from radiomics import featureextractor
import warnings

from transformers.models.udop.convert_udop_to_hf import original_transform

warnings.filterwarnings('ignore')
import logging
import pickle
logging.disable()

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Reshape, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical


def normalize_csv(input_file, output_file, metapath):
    df = pd.read_csv(input_file)
    meta_df = pd.read_csv(metapath)
    scaler = MinMaxScaler()

    exclude_cols = ['id', 'file_name']
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if col not in exclude_cols]

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    class_dict = {}
    for _, row in meta_df.iterrows():
        file_name = str(row['img_path']).split("/")[-1]
        class_dict[file_name] = row['class']
    
    df['category'] = df['file_name'].map(class_dict)

    df.to_csv(output_file, index=False)
    print(f"Normalized file saved to: {output_file}")

def merge_and_filter_csv(folder_path, output_file, category_col='category', threshold=0.05):
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]

    common_columns = None
    dfs = []

    for file_path in csv_files:
        df = pd.read_csv(file_path)
        df[category_col] = os.path.splitext(os.path.basename(file_path))[0]
        dfs.append(df)
        if common_columns is None:
            common_columns = set(df.columns)
        else:
            common_columns.intersection_update(df.columns)

    if not common_columns:
        raise ValueError("No common columns found, cannot merge files.")

    common_columns = list(common_columns)
    merged_df = pd.concat([df[common_columns] for df in dfs], ignore_index=True)

    if merged_df[category_col].dtype == 'object':
        merged_df[category_col] = merged_df[category_col].astype('category').cat.codes
    correlations = merged_df.corr()[category_col].abs()

    low_corr_cols = correlations[correlations < threshold].index

    filtered_df = merged_df.drop(columns=low_corr_cols)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    filtered_df.to_csv(output_file, index=False)
    print(f"Merged and filtered file saved to: {output_file}")

def filter_low_variance(input_file, output_file, threshold=0.95):
    df = pd.read_csv(input_file)
    
    exclude_cols = ['id', 'file_name']
    numeric_cols = [col for col in df.columns if col not in exclude_cols]
    
    corr_matrix = df[numeric_cols].corr()
    
    to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                to_drop.add(corr_matrix.columns[j])
    
    df_filtered = df.drop(columns=list(to_drop))
    
    df_filtered.to_csv(output_file, index=False)
    print(f"Filtered file saved to: {output_file}")

def process_csv_folder(folder_path,metapath , threshold=0.95):
    original_folder = os.path.join(folder_path,"1.Original")
    normalized_path = os.path.join(folder_path, "2.Normalized.csv")
    filtered_path = os.path.join(folder_path, "3.Filtered.csv")
    merged_path = os.path.join(folder_path, "4.Merged.csv")
    finial_path = os.path.join(folder_path, "5.Finial.csv")

    input_path = os.path.join(original_folder, "radiomics.csv")
    print("meta ",metapath)
    normalize_csv(input_path, normalized_path,metapath)

    train_classifier(normalized_path, finial_path)



def extra_main(root_path:str,img_folder_name,seg_folder_name):
    params = 'feature_extract_setting.yaml'
    extractor = featureextractor.RadiomicsFeatureExtractor(params)
    features = {}
    out_folder_path = os.path.join(root_path, "radiomics", "1.Original")


    if img_folder_name is not None and seg_folder_name is not None:
        images_path = os.path.join(root_path,img_folder_name)
        labels_path = os.path.join(root_path,seg_folder_name)
    else:
        images_path = os.path.join(root_path,"HAM10000_images")
        labels_path = os.path.join(root_path,"HAM10000_GroundTruth")

    img_file_name_list = os.listdir(images_path)
    print("Total file count", len(img_file_name_list))
    bar = tqdm(enumerate(img_file_name_list), total=len(img_file_name_list))
    for img_file_name_index, img_file_name in bar:
        if "_superpixels" in img_file_name:
            continue
        if ".png" not in img_file_name and ".jpg" not in img_file_name:
            continue
        if str(img_file_name) in features:
            continue
        if img_file_name.find("DS_Store") >= 0:
            continue
        image = sitk.ReadImage(os.path.join(images_path ,img_file_name), sitk.sitkUInt32)

        label_file_name = img_file_name.replace('.jpg','_segmentation.png')
        label = sitk.ReadImage(os.path.join(labels_path, label_file_name), sitk.sitkUInt32)
        try:
            label_num = int(np.unique(sitk.GetArrayFromImage(label))[1:][-1])
        except Exception as e:
            print(f"Exception: {e}")
            label_num = 255
        bar.set_description("Processing {0}".format(img_file_name)+",label_num="+str(label_num))

        features[str(img_file_name)] = extractor.execute(image, label, label=label_num)

    feature_names = list(
        sorted(filter(lambda k: not k.startswith("diagnostics_"), features.get(img_file_name_list[0]))))
    if not os.path.exists(out_folder_path):
        os.makedirs(out_folder_path)

    out_file_path = os.path.join(out_folder_path,"radiomics.csv")
    if os.path.exists(out_file_path):
        os.remove(out_file_path)
    f = open(out_file_path, 'a')
    f.write("id,file_name")
    for feature_names_item in feature_names:
        f.write("," + feature_names_item)
    f.write("\n")
    id = 0
    for file_name in features:
        f.write(str(id) + "," + str(file_name))
        id += 1
        for feature_names_item in feature_names:
            f.write("," + str(features.get(file_name)[feature_names_item]))
            f.flush()
        f.write("\n")
    f.close()
    print("Radiomics extraction completed.")

def train_classifier(normalized_path, final_path):
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("Using GPU for training")
    else:
        print("No GPU found, using CPU")
    
    df = pd.read_csv(normalized_path)
    
    exclude_cols = ['id', 'file_name']
    feature_cols = [col for col in df.columns if col not in exclude_cols and col != 'category']
    
    X = df[feature_cols].values
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['category'])
    num_classes = len(np.unique(y))
    y = to_categorical(y)
    
    n_features = len(feature_cols)
    X = X.reshape((X.shape[0], n_features, 1))
    
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_features, 1)),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(1000, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=50,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    model.fit(X, y, epochs=1000, batch_size=32, validation_split=0.2, callbacks=callbacks)
    
    intermediate_layer_model = Sequential(model.layers[:-1])
    intermediate_output = intermediate_layer_model.predict(X)
    
    max_indices = np.argmax(intermediate_output, axis=1)
    
    df['feature_class'] = max_indices
    
    df.to_csv(final_path, index=False)
    print(f"Classification results saved to: {final_path}")