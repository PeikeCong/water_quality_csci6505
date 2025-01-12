import pickle
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


RAW_PATH = "./Data/water_quality"
PATH_DROP = "./Data/water_quality_drop"
PATH_KNN = "./Data/water_quality_KNN"
PATH_KNN_PCA = "./Data/water_quality_KNN_PCA"
PATH_MINI = "./Data/water_quality_mini"
PATH_MINI_KNN = "./Data/water_quality_min_KNN"
PATH_MINI_KNN_PCA = "./Data/water_quality_min_KNN_PCA"

PATH_KNN_FEATURES = "./Data/water_quality_FEATURES"
PATH_MINI_KNN_FEATURES = "./Data/water_quality_mini_FEATURES"


def save_data(path, data):
    with open(path, "wb") as file:
        pickle.dump(data, file)


def read_data(path):
    with open(path, "rb") as file:
        data = pickle.load(file)
    return data


def data_info():
    water_quality = pd.read_csv('./Data/brisbane_water_quality.csv')

    print(f"Total samples: {water_quality.shape[0]}")
    print()
    print(f"Columns info: ")
    print(water_quality.columns)
    print()
    print(f"missing values: ")
    print(water_quality.isnull().sum())

    # delete unwanted columns
    water_quality = water_quality.drop(['Record number',
                                        'Chlorophyll [quality]',
                                        'Temperature [quality]',
                                        'Dissolved Oxygen [quality]',
                                        'Dissolved Oxygen (%Saturation)',
                                        'Dissolved Oxygen (%Saturation) [quality]',
                                        'pH [quality]',
                                        'Salinity [quality]',
                                        'Specific Conductance [quality]',
                                        'Turbidity [quality]'], axis=1)
    print(water_quality.columns)
    save_data(RAW_PATH, water_quality)


def address_timestamp():
    water_quality = read_data(RAW_PATH)
    # convert to time stamp
    water_quality['Timestamp'] = pd.to_datetime(water_quality['Timestamp'])
    water_quality = water_quality.sort_values('Timestamp')
    water_quality['Timestamp'] = water_quality['Timestamp'].astype('int64') // 10**9

    # delete dumplicate
    water_quality = water_quality.drop_duplicates(subset='Timestamp', keep='first').reset_index(drop=True)
    # check dumplicate
    duplicates = water_quality[water_quality['Timestamp'].duplicated()]
    if not duplicates.empty:
        print(duplicates)
    else:
        print("no duplicate")
    print(f"Total samples: {water_quality.shape[0]}")

    print("Time Difference")
    time_diffs = water_quality['Timestamp'].diff()
    time_diffs = time_diffs.dropna()
    counts = time_diffs.value_counts()
    print(counts)
    # add timediff as new feature
    water_quality['TimeDiff'] = water_quality['Timestamp'].diff().fillna(0)
    water_quality['TimeDiff'] = water_quality['TimeDiff'].astype('int64')
    water_quality.insert(1, 'TimeDiff', water_quality.pop('TimeDiff'))
    print(water_quality.dtypes)
    save_data(RAW_PATH, water_quality)


def address_timestamp_mini_sample():
    water_quality = read_data(RAW_PATH)
    # convert to time stamp
    water_quality['Timestamp'] = pd.to_datetime(water_quality['Timestamp'])
    water_quality = water_quality.sort_values('Timestamp')
    water_quality['Timestamp'] = water_quality['Timestamp'].astype('int64') // 10**9

    # delete dumplicate
    water_quality = water_quality.drop_duplicates(subset='Timestamp', keep='first').reset_index(drop=True)
    # check dumplicate
    duplicates = water_quality[water_quality['Timestamp'].duplicated()]
    if not duplicates.empty:
        print(duplicates)
    else:
        print("no duplicate")
    print(f"Total samples: {water_quality.shape[0]}")

    print("Time Difference")
    time_diffs = water_quality['Timestamp'].diff()
    time_diffs = time_diffs.dropna()
    counts = time_diffs.value_counts()
    print(counts)
    sample_indices = time_diffs[time_diffs == 600].index
    print(sample_indices[0])
    print(sample_indices[-1])

    water_quality = water_quality.iloc[5830:30613]
    water_quality = water_quality.reset_index(drop=True)
    time_diffs = water_quality['Timestamp'].diff()
    time_diffs = time_diffs.dropna()
    counts = time_diffs.value_counts()
    print(counts)
    sample_indices = time_diffs[time_diffs == 297600].index
    print(sample_indices)
    sample_indices = time_diffs[time_diffs == 2631000].index
    print(sample_indices)

    print(len(water_quality))
    water_quality = water_quality.iloc[0:18579]
    time_diffs = water_quality['Timestamp'].diff()
    time_diffs = time_diffs.dropna()
    counts = time_diffs.value_counts()
    print(counts)
    print(len(water_quality))

    # 18579
    save_data(PATH_MINI, water_quality)



def missing_values_drop():
    water_quality = read_data(RAW_PATH)
    water_quality.dropna(inplace=True)
    print(f"Total sample: {water_quality.shape[0]}")
    save_data(PATH_DROP, water_quality)


def missing_values_KNN():
    # water_quality = read_data(RAW_PATH)
    water_quality = read_data(PATH_MINI)
    imputer = KNNImputer(n_neighbors=5)
    water_quality_imputed = pd.DataFrame(imputer.fit_transform(water_quality.select_dtypes(include=['float64', 'int64'])),
                                        columns = water_quality.select_dtypes(include=['float64', 'int64']).columns)
    water_quality[water_quality_imputed.columns] = water_quality_imputed
    print(f"Missing values: ")
    print(water_quality.isnull().sum())
    # save_data(PATH_KNN, water_quality)
    save_data(PATH_MINI_KNN, water_quality)



def feature_engineering():
    water_quality = read_data(PATH_KNN)
    # normalization
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(water_quality)
    scaled_data = pd.DataFrame(scaled_data, columns=water_quality.columns)
    time_features = scaled_data.iloc[:, :2]
    quality_features = scaled_data.iloc[:, 2:]

    # PCA
    pca = PCA()
    pca_data = pca.fit_transform(quality_features)

    print("The Composition of each PC:")
    pc_components = pd.DataFrame(pca.components_, columns=quality_features.columns, 
                                 index=[f'PC{i+1}' for i in range(pca.n_components_)])
    print(pc_components)
    print()

    explained_variance = pca.explained_variance_ratio_
    print("Explained variance ratio of each principal component:")
    for i, ratio in enumerate(explained_variance):
        print(f"Principal Component {i + 1}: {ratio:.2%}")
    print()

    n_components = np.argmax(np.cumsum(explained_variance) >= 0.95) + 1
    print(f"Number of components to retain 95% variance: {n_components}")
    print()

    pca_optimal = PCA(n_components=n_components)
    reduced_data = pca_optimal.fit_transform(scaled_data)
    pca_df = pd.DataFrame(reduced_data, columns=[f'PC{i+1}' for i in range(n_components)])
    print(pca_df)

    combined_data = pd.concat([time_features, pca_df], axis=1)
    print(combined_data)
    save_data(PATH_KNN_PCA, combined_data)


def each_features_for_min():

    water_quality = read_data(PATH_MINI_KNN)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(water_quality)
    scaled_data = pd.DataFrame(scaled_data, columns=water_quality.columns)

    time_features = scaled_data.iloc[:, :1]
    non_time_features = scaled_data.iloc[:, 1:]
    X = non_time_features.drop(columns=["Dissolved Oxygen"]) 
    Y = non_time_features["Dissolved Oxygen"]                

    print("Time features:")
    print(time_features.head())
    print()
    print("Feature data (X):")
    print(X.head())
    print()
    print("Target variable (Y):")
    print(Y.head())
    print()

    combined_data = pd.concat([time_features, X, Y], axis=1)  
    save_data(PATH_MINI_KNN_FEATURES, combined_data)        



def each_features():
    water_quality = read_data(PATH_KNN)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(water_quality)
    scaled_data = pd.DataFrame(scaled_data, columns=water_quality.columns)
    
    X = scaled_data.drop(columns=["Dissolved Oxygen"])
    Y = scaled_data["Dissolved Oxygen"]

    print("Feature data (X):")
    print(X.head())
    print()
    print("Target variable (Y):")
    print(Y.head())
    print()

    combined_data = pd.concat([X, Y], axis=1)
    print(combined_data)
    save_data(PATH_KNN_FEATURES, combined_data)



if __name__ == "__main__":
    # data_info()
    # address_timestamp()
    # missing_values_drop()
    # missing_values_KNN()
    # feature_engineering()
    # address_timestamp_mini_sample()
    # missing_values_KNN()
    # feature_engineering_for_min()

    each_features()
    each_features_for_min()
