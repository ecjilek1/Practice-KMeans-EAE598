# Practice-KMeans-EAE598
**Project Overview with K-Means Clustering!**

1. Preprocessing: Preparing for K-Means Clustering

2. Applying K-Means

3. Unit Testing: How Well did the Model Perform

**Preprocessing: Preparing for K-Means Clustering**

Data was downloaded from the Climate Data Store under ERA5 Hourly Reanalysis Data. This data is available for anyone!

The tools utilized in this analysis are (remove # as needed)
```
#!pip install matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import BoundaryNorm, ListedColormap
#!pip install pandas
import pandas as pd
#!pip install numpy
import numpy as np
#!pip install xarray
import xarray as xr
#!pip install minisom
#!pip install netCDF4
#!apt-get install libproj-dev proj-data proj-bin libgeos-dev
!pip install Cartopy
import cartopy.crs as ccrs
import cartopy.crs as ccrs
import cartopy.feature as cfeature
!pip install basemap
from mpl_toolkits.basemap import Basemap
import os
import calendar
```
Specifically for K-Means
```
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
```
**Importing Data**
```
from google.colab import drive
drive.mount('/content/drive')
```
```
folder_path = 'YOUR FILES'
```
```
datasets = []

for filename in sorted(os.listdir(folder_path)):
    if filename.endswith('.nc'):
        file_path = os.path.join(folder_path, filename)
        dataset = xr.open_dataset(file_path)
        datasets.append(dataset)
        print(f"Loaded {filename}")

print(f"\nTotal files loaded : {len(datasets)}")
```
```
for i, ds in enumerate(datasets):
    times = pd.to_datetime(ds['valid_time'].values)
    years = sorted(set(times.year))

    if len(years) == 1:
        print(f"Dataset {i}: Year = {years[0]}")
    else:
        print(f"Dataset {i}: Years = {years}")
```
```
named_datasets = {
    "1940": datasets[4],
    "1954": datasets[11],
    "1956": datasets[6],
    "1980": datasets[12],
    "1988": datasets[0],
    "1999": datasets[8],
    "2002": datasets[5],
    "2003": datasets[3],
    "2006": datasets[13],
    "2007": datasets[7],
    "2011": datasets[9],
    "2012": datasets[10],
    "2018": datasets[2],
    "2020": datasets[14],
    "2023": datasets[1],
}

ds06 = named_datasets["2006"]
ds40 = named_datasets["1940"]
ds54 = named_datasets["1954"]
ds56 = named_datasets["1956"]
ds80 = named_datasets["1980"]
ds88 = named_datasets["1988"]
ds99 = named_datasets["1999"]
ds02 = named_datasets["2002"]
ds03 = named_datasets["2003"]
ds07 = named_datasets["2007"]
ds11 = named_datasets["2011"]
ds12 = named_datasets["2012"]
ds18 = named_datasets["2018"]
ds20 = named_datasets["2020"]
ds23 = named_datasets["2023"]
```
```
# 2006
ds06 = named_datasets["2006"]

# Get the first time step
time = ds06.coords['valid_time'][0]

# Extract PV and Z at 500 hPa
pv06 = ds06['pv'].sel(valid_time=time, pressure_level=500.0) / 1e-6  # Convert to PVU
z06 = ds06['z'].sel(valid_time=time, pressure_level=500.0) / 9.80665  # Convert to meters
```
```
# 1940
ds40 = named_datasets["1940"]

# Get the first time step
time = ds40.coords['valid_time'][0]

# Extract PV and Z at 500 hPa
pv40 = ds40['pv'].sel(valid_time=time, pressure_level=500.0) / 1e-6  # Convert to PVU
z40 = ds40['z'].sel(valid_time=time, pressure_level=500.0) / 9.80665  # Convert to meters
```
```
# 1954
ds54 = named_datasets["1954"]

# Get the first time step
time = ds54.coords['valid_time'][0]

# Extract PV and Z at 500 hPa
pv54 = ds54['pv'].sel(valid_time=time, pressure_level=500.0) / 1e-6  # Convert to PVU
z54 = ds54['z'].sel(valid_time=time, pressure_level=500.0) / 9.80665  # Convert to meters
```
```
# 1956
ds56 = named_datasets["1956"]

# Get the first time step
time = ds56.coords['valid_time'][0]

# Extract PV and Z at 500 hPa
pv56 = ds56['pv'].sel(valid_time=time, pressure_level=500.0) / 1e-6  # Convert to PVU
z56 = ds56['z'].sel(valid_time=time, pressure_level=500.0) / 9.80665  # Convert to meters
```
```
# 1980
ds80 = named_datasets["1980"]

# Get the first time step
time = ds80.coords['valid_time'][0]

# Extract PV and Z at 500 hPa
pv80 = ds80['pv'].sel(valid_time=time, pressure_level=500.0) / 1e-6  # Convert to PVU
z80 = ds80['z'].sel(valid_time=time, pressure_level=500.0) / 9.80665  # Convert to meters
```
```
# 1988
ds88 = named_datasets["1988"]

# Get the first time step
time = ds88.coords['valid_time'][0]

# Extract PV and Z at 500 hPa
pv88 = ds88['pv'].sel(valid_time=time, pressure_level=500.0) / 1e-6  # Convert to PVU
z88 = ds88['z'].sel(valid_time=time, pressure_level=500.0) / 9.80665  # Convert to meters
```
```
# 1999
ds99 = named_datasets["1999"]

# Get the first time step
time = ds99.coords['valid_time'][0]

# Extract PV and Z at 500 hPa
pv99 = ds99['pv'].sel(valid_time=time, pressure_level=500.0) / 1e-6  # Convert to PVU
z99 = ds99['z'].sel(valid_time=time, pressure_level=500.0) / 9.80665  # Convert to meters
```
```
# 2002
ds02 = named_datasets["2002"]

# Get the first time step
time = ds02.coords['valid_time'][0]

# Extract PV and Z at 500 hPa
pv02 = ds02['pv'].sel(valid_time=time, pressure_level=500.0) / 1e-6  # Convert to PVU
z02 = ds02['z'].sel(valid_time=time, pressure_level=500.0) / 9.80665  # Convert to meters
```
```
# 2003
ds03 = named_datasets["2003"]

# Get the first time step
time = ds03.coords['valid_time'][0]

# Extract PV and Z at 500 hPa
pv03 = ds03['pv'].sel(valid_time=time, pressure_level=500.0) / 1e-6  # Convert to PVU
z03 = ds03['z'].sel(valid_time=time, pressure_level=500.0) / 9.80665  # Convert to meters
```
```
# 2007
ds07 = named_datasets["2007"]

# Get the first time step
time = ds07.coords['valid_time'][0]

# Extract PV and Z at 500 hPa
pv07 = ds07['pv'].sel(valid_time=time, pressure_level=500.0) / 1e-6  # Convert to PVU
z07 = ds07['z'].sel(valid_time=time, pressure_level=500.0) / 9.80665  # Convert to meters
```
```
# 2011
ds11 = named_datasets["2011"]

# Get the first time step
time = ds11.coords['valid_time'][0]

# Extract PV and Z at 500 hPa
pv11 = ds11['pv'].sel(valid_time=time, pressure_level=500.0) / 1e-6  # Convert to PVU
z11 = ds11['z'].sel(valid_time=time, pressure_level=500.0) / 9.80665  # Convert to meters
```
```
# 2012
ds12 = named_datasets["2012"]

# Get the first time step
time = ds12.coords['valid_time'][0]

# Extract PV and Z at 500 hPa
pv12 = ds12['pv'].sel(valid_time=time, pressure_level=500.0) / 1e-6  # Convert to PVU
z12 = ds12['z'].sel(valid_time=time, pressure_level=500.0) / 9.80665  # Convert to meters
```
```
# 2018
ds18 = named_datasets["2018"]

# Get the first time step
time = ds18.coords['valid_time'][0]

# Extract PV and Z at 500 hPa
pv18 = ds18['pv'].sel(valid_time=time, pressure_level=500.0) / 1e-6  # Convert to PVU
z18 = ds18['z'].sel(valid_time=time, pressure_level=500.0) / 9.80665  # Convert to meters
```
```
# 2020
ds20 = named_datasets["2020"]

# Get the first time step
time = ds20.coords['valid_time'][0]

# Extract PV and Z at 500 hPa
pv20 = ds20['pv'].sel(valid_time=time, pressure_level=500.0) / 1e-6  # Convert to PVU
z20 = ds20['z'].sel(valid_time=time, pressure_level=500.0) / 9.80665  # Convert to meters
```
```
# 2023
ds23 = named_datasets["2023"]

# Get the first time step
time = ds23.coords['valid_time'][0]

# Extract PV and Z at 500 hPa
pv23 = ds23['pv'].sel(valid_time=time, pressure_level=500.0) / 1e-6  # Convert to PVU
z23 = ds23['z'].sel(valid_time=time, pressure_level=500.0) / 9.80665  # Convert to meters
```
```
lat = ds['latitude'].values
lon = ds['longitude'].values
lon2d, lat2d = np.meshgrid(lon, lat)
```

**Setting Up the Area**
```
fig, ax = plt.subplots(figsize=(8, 6))

m = Basemap(projection='cyl',
            llcrnrlon=-105, llcrnrlat=23,
            urcrnrlon=-87, urcrnrlat=50,
            resolution='i')

m.drawcoastlines()
m.drawcountries()
m.drawstates()


parallels = np.arange(23, 50, 3)
m.drawparallels(parallels, labels=[1, 0, 0, 0], linewidth=0.5)

meridians = np.arange(-105, -87, 3)
m.drawmeridians(meridians, labels=[0, 0, 0, 1], linewidth=0.5)
```
![image](https://github.com/user-attachments/assets/5ab8f35a-be25-4580-84d6-115afdb5a5a9)

**Setting Up the Custom Color Bar**
```
colors = [
    'black', 'dimgray',
    'gray', 'darkgrey',
    'lightgrey', 'white',
    'white', 'lightyellow',
    'yellow', 'gold',
    'orange', 'orangered',
    'red', 'deeppink',
    'magenta', 'mediumvioletred',
    'purple', 'darkblue'
]
boundaries = [-5, -3, -2.5, -2, -1, -0.5, -0.1, 0.1, 0.5, 1,
              1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6]

pv_cmap = ListedColormap(colors)
pv_norm = BoundaryNorm(boundaries, ncolors=len(colors))
```

**Example Year: Checking the Code with Adding Potential Vorticity (PV) and Geopotential Height (Z)**
```
ds = ds20

year = 2020
month = 7
day = 31
hour = 0

target_time = pd.Timestamp(f"{year}-{month:02d}-{day:02d}T{hour:02d}:00")
time = ds.sel(valid_time=target_time, method='nearest')['valid_time']
print(f"Using time: {time.values}")

pv = ds['pv'].sel(valid_time=time, pressure_level=500.0) / 1e-6  # Convert to PVU
z = ds['z'].sel(valid_time=time, pressure_level=500.0) / 9.80665  # Convert to meters

lat = ds['latitude'].values
lon = ds['longitude'].values
lon = np.where(lon > 180, lon - 360, lon)  # Convert to -180 to 180 if needed
lon2d, lat2d = np.meshgrid(lon, lat)

fig, ax = plt.subplots(figsize=(10, 6))
m = Basemap(projection='cyl',
            llcrnrlon=-105, llcrnrlat=23,
            urcrnrlon=-87, urcrnrlat=50,
            resolution='i', ax=ax)

m.drawcoastlines()
m.drawcountries()
m.drawstates()
m.drawparallels(np.arange(np.floor(lat.min()), np.ceil(lat.max()) + 1, 5),
                labels=[1,0,0,0], linewidth=0.5)
m.drawmeridians(np.arange(np.floor(lon.min()), np.ceil(lon.max()) + 1, 5),
                labels=[0,0,0,1], linewidth=0.5)

# Plot PV
cf = m.contourf(lon2d, lat2d, pv, levels=boundaries, cmap=pv_cmap, norm=pv_norm, latlon=True)
cb = m.colorbar(cf, location='right', pad=0.03, ticks=boundaries)
cb.set_label('PVU')

# Plot geopotential height
cs = m.contour(lon2d, lat2d, z, levels=np.arange(5400, 6000, 20), colors='black', linewidths=1, latlon=True)
plt.clabel(cs, fmt='%d', fontsize=8)

plt.title(f'ERA5 PV and Geopotential Height at 500 hPa\n{str(time.values)[:16]}')
plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/3622fae9-d8b6-4c9c-9b4e-e0fef66fe5d7)

**How to Save Data: Looping through Hour Data**
```

base_output_folder = '/content/drive/MyDrive/ERA5 598 Data/2023/Both'

ds = ds23  
times = pd.to_datetime(ds['valid_time'].values)

unique_months = sorted(set((t.year, t.month) for t in times))

for year, month in unique_months:
    print(f"\n Processing {year}-{month:02d}...")

    # Create subfolder for month
    output_folder = os.path.join(base_output_folder, f"{year}_{month:02d}")
    os.makedirs(output_folder, exist_ok=True)

    # Get last day of the month
    last_day = calendar.monthrange(year, month)[1]

    # Time slice for the month
    all_times = ds['valid_time'].sel(valid_time=slice(
        f'{year}-{month:02d}-01', f'{year}-{month:02d}-{last_day:02d}'))

    # Filter every 6 hours
    times_6hr = all_times[::6]

    for t in times_6hr:
        time = pd.to_datetime(t.values)
        print(f"Plotting: {time}")

        # Data
        pv = ds['pv'].sel(valid_time=t, pressure_level=500.0) / 1e-6
        z = ds['z'].sel(valid_time=t, pressure_level=500.0) / 9.80665

        lat = ds['latitude'].values
        lon = ds['longitude'].values
        lon = np.where(lon > 180, lon - 360, lon)
        lon2d, lat2d = np.meshgrid(lon, lat)

      
        cf = m.contourf(lon2d, lat2d, pv, levels=boundaries,
                        cmap=pv_cmap, norm=pv_norm, latlon=True)
        cb = m.colorbar(cf, location='right', pad=0.03, ticks=boundaries)
        cb.set_label('PVU')

        cs = m.contour(lon2d, lat2d, z, levels=np.arange(5400, 6000, 20),
                       colors='black', linewidths=1, latlon=True)
        plt.clabel(cs, fmt='%d', fontsize=8)

        # Save
        plt.title(f'ERA5 PV and Geopotential Height at 500 hPa\n{time:%Y-%m-%d %H:%M UTC}')
        plt.tight_layout()
        filename = f"PV_Z_500hPa_{time:%Y%m%d_%H%M}.png"
        filepath = os.path.join(output_folder, filename)
        plt.savefig(filepath, dpi=150)
        plt.close()

print("\nAll monthly plots saved")
```
**K-Means Testing!**
```
#Folder with all created maps
folder_path = 'YOUR FOLDER WITH ALL IMAGES'
```
```
image_paths = [os.path.join(folder_path, f)
  for f in os.listdir(folder_path) if f.lower().endswith(('.png'))]
print(f"Found {len(image_paths)} images.")
#There should be a total of 3,300+ images
```
```
def preprocess_image(image_path, target_size=(64, 64)):
    img = Image.open(image_path).convert('RGB').resize(target_size)
    return np.array(img).flatten()
```
```
image_data = []
for path in image_paths:
    img_vector = preprocess_image(path)
    image_data.append(img_vector)

image_data = np.array(image_data)
print(f"Image data shape: {image_data.shape}")
```
```
n_clusters = 8
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(image_data)

print(f"KMeans clustering completed. Cluster labels: {labels}")
```
```
max_images_per_cluster = 5
fig, axes = plt.subplots(n_clusters, max_images_per_cluster, figsize=(35, n_clusters * 3))

for cluster in range(n_clusters):
    cluster_indices = np.where(labels == cluster)[0]
    selected_indices = cluster_indices[:max_images_per_cluster]

    for i in range(max_images_per_cluster):
        ax = axes[cluster, i]
        ax.axis('off')

        if i < len(selected_indices):
            img = Image.open(image_paths[selected_indices[i]])
            ax.imshow(img)

    axes[cluster, 0].set_title(f'Cluster {cluster}', loc='left', fontsize=14, color='blue')

plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/ea10531c-aacc-4610-8382-b5e8e187d202)

```
cluster_range = range(2,15)
inertias = []

for k in cluster_range:
  print (f"Fitting Kmeans for {k} clusters")
  kmeans_test = KMeans(n_clusters=k, random_state=42)
  kmeans_test.fit(image_data)
  inertias.append(kmeans_test.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(cluster_range, inertias, marker='o', linestyle='-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method Clusters')
plt.xticks(cluster_range)
plt.grid(True)
plt.show()
```
![image](https://github.com/user-attachments/assets/20f87c39-d335-4c26-9818-d949613fad34)
```
from sklearn.metrics import davies_bouldin_score

cluster_range = range(2, 15)
dbi_scores = []

for k in cluster_range:
    print(f"Fitting KMeans for {k} clusters")
    kmeans_test = KMeans(n_clusters=k, random_state=42)
    labels_test = kmeans_test.fit_predict(image_data)
    dbi = davies_bouldin_score(image_data, labels_test)
    dbi_scores.append(dbi)

plt.figure(figsize=(8, 5))
plt.plot(cluster_range, dbi_scores, marker='o', linestyle='-', color='green')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Davies-Bouldin Index')
plt.title('Davies-Bouldin Index for KMeans Clustering')
plt.xticks(cluster_range)
plt.grid(True)
plt.show()
```
![image](https://github.com/user-attachments/assets/84a47209-08ef-4f42-92fb-902bbf0400c8)
```
from sklearn.metrics import silhouette_score
score = silhouette_score(image_data, kmeans.labels_)
print(f"Silhouette Score: {score:.3f}")
#Originally, got a low score of 0.024
```
<img width="211" alt="Screenshot 2025-05-03 at 15 29 54" src="https://github.com/user-attachments/assets/7d41bfe7-edc1-43a5-b6fd-03ad6aed3751" />

**Additional Cluster Testing!**
```
from sklearn.metrics import calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
```
```
scaled_data = StandardScaler().fit_transform(image_data)

ch_scores = []
cluster_range = range(2, 11)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(scaled_data)
    score = calinski_harabasz_score(scaled_data, labels)
    ch_scores.append(score)

best_k = cluster_range[np.argmax(ch_scores)]

final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
final_labels = final_kmeans.fit_predict(scaled_data)
print(f"k={k}: CH Score = {score:.2f}")
#with k=10, CH Score is 23.65
```
<img width="214" alt="Screenshot 2025-05-03 at 15 27 45" src="https://github.com/user-attachments/assets/f6a4f80a-ea9f-4c59-be09-adaf9eba963d" />

```
from sklearn.manifold import TSNE
```
```
custom_k = 10 
kmeans = KMeans(n_clusters=custom_k, random_state=42, n_init='auto')
final_labels = kmeans.fit_predict(scaled_data)

tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(scaled_data)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=final_labels, cmap='tab10', s=20)
plt.colorbar(scatter, label='Cluster Label')
plt.title(f't-SNE Visualization of KMeans Clusters (k={custom_k})')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.grid(True)
plt.show()
```
![image](https://github.com/user-attachments/assets/0288f5b2-137d-4f14-b32a-1c626c97dc4f)

**Let's Try Using ResNet50 along with CNN!**
```
#Batch Size is 32
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
class ImageFolderDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), path

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]
dataset = ImageFolderDataset(image_paths, transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))  
model.to(device)
model.eval()

features = []
paths = []

with torch.no_grad():
    for batch_imgs, batch_paths in tqdm(loader):
        batch_imgs = batch_imgs.to(device)
        batch_feats = model(batch_imgs).squeeze()  
        features.append(batch_feats.cpu().numpy())
        paths.extend(batch_paths)

features = np.vstack(features)
print(f"Features shape: {features.shape}")  
```
```
#Changing K-Means Clustering to n_clusters to 20!
from sklearn.cluster import KMeans

n_clusters =20
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(features)
```
```
from sklearn.metrics import silhouette_score
score = silhouette_score(features, labels)
print(f"Silhouette Score (CNN features): {score:.3f}")
#Score with CNN is 0.094
```
<img width="339" alt="Screenshot 2025-05-03 at 15 28 42" src="https://github.com/user-attachments/assets/47608b28-2d30-4a74-9bb9-3f8cbe1a5d2d" />
