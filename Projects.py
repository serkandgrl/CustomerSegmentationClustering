import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None) #* -->Kolon gösterim sınırını kaldırdık
pd.set_option("display.width", 500) #* --> #* -->Genişlik ayarladık

df = pd.read_csv("D:\DataAnalysis\CustomerSegmentationClustering\Mall_Customers.csv") #* --> veri setini okuduk

def check_df(dataframe, head = 5):
    print("################### Shape ##################")
    print(dataframe.shape)
    print("################### Types ##################")
    print(dataframe.dtypes)
    print("################### Head ##################")
    print(dataframe.head(head))
    print("################### Tail ##################")
    print(dataframe.tail(head))
    print("################### NA ##################")
    print(dataframe.isnull().sum())
    print("################### Quantiles ##################")
    print(dataframe.describe([0, 0.05, 0.50, 0.75]).T)

check_df(df) #* -->Veri setine genel bir bakış

def grab_col_names(dataframe, cat_th = 5, car_th = 10):
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ["int", "float"]]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df) #* -->Sütunları kategorik, sayısal ve karmaşık kategorik olarak ayırır ve bu sütunları değişkenlere atar.

def cat_summary(dataframe, col_name, plot = False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("--------------------------------------------------")
    if plot:
        sns.countplot(x = dataframe[col_name], data = dataframe)
        plt.show(block = True)
#* -->Bu fonksiyon, kategorik bir sütunun değerlerini ve oranlarını gösterir ve kategorik sütunun değerlerinin countplot grafiğini çizer.

for col in cat_cols:
    cat_summary(df, col, plot = True)

def num_summary(dataframe, numerical_col, plot = False):
    print(dataframe[numerical_col].describe().T)
    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block = True)
#* -->Bu fonksiyon, sayısal bir sütunun istatistiksel özetini gösterir ve sayısal sütunun histogram grafiğini çizer.

for col in num_cols:
    num_summary(df, col, plot = True)

#Univariate Analysis --> Tek değişkenli analiz

df.describe().T
columns = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]

for col in columns:
    plt.figure()
    sns.distplot(df[col])
#* -->Bu blok, belirli sayısal sütunlar için distplot grafiği çizer.

for col in columns:
    plt.figure()
    sns.kdeplot(data = df, x = col, hue = "Gender", shade = True)
    plt.show()
#* -->Bu blok, belirli sayısal sütunlar için kdeplot grafiği çizer ve cinsiyete göre renklendirir.

for col in columns:
    plt.figure()
    sns.boxplot(data = df, x = "Gender", y = df[col])
#* -->Bu blok, belirli sayısal sütunlar için boxplot grafiği çizer ve cinsiyete göre gruplandırır.

df["Gender"].value_counts(normalize = True) #* -->sütunuda bulunan değerlerin oranlarını gösterir.

#Bivariate Analysis --> İki değişken arasındaki analiz

sns.scatterplot(data = df, x = "Annual Income (k$)", y = "Spending Score (1-100)") #* -->Kolonlar arasındaki ilişkiyi scatterplot ile çizdik
df = df.drop("CustomerID", axis = 1)
sns.pairplot(df, hue = "Gender")
df.groupby(["Gender"])[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].mean()

numeric_columns = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
df_numeric = df[numeric_columns]
correlation_matrix = df_numeric.corr()
sns.heatmap(correlation_matrix, annot = True, cmap = "coolwarm")
#* -->Bu blok, sayısal sütunlar arasındaki korelasyon matrisini hesaplar ve ısı haritası olarak görselleştirir.

#Clustering - Univariate, Bivariate, Multivariate

clustering1 = KMeans(n_clusters = 3)
clustering1.fit(df[["Annual Income (k$)"]])
df["Income Cluster"] = clustering1.labels_
df["Income Cluster"].value_counts()
clustering1.inertia_
#* -->Bu blok, "Annual Income (k$)" sütunu için K-Means kümeleme algoritması uygular.3 küme oluşturur ve her veriyi bir kümeye atar."Income Cluster" adlı yeni bir sütun oluşturur ve her verinin sayısını döndürür

inertia_score = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df[["Annual Income (k$)"]])
    inertia_score.append(kmeans.inertia_)
plt.plot(range(1, 11), inertia_score)
#* -->Bu blok, farklı küme sayıları için K-Means algoritmasının inertia değerini hesaplar ve bu değerleri bir grafik üzerinde gösterir.Bu grafik, küme sayısını belirlemek için kullanılabilir.

df.groupby("Income Cluster")[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].mean()

#Bivariate Clustering

clustering2 = KMeans(n_clusters = 5)
clustering2.fit(df[["Annual Income (k$)", "Spending Score (1-100)"]])
df["Spending and Income Cluster"] = clustering2.labels_
inertia_score2 = []

for i in range(1, 11):
    kmeans2 = KMeans(n_clusters = i)
    kmeans2.fit(df[["Annual Income (k$)", "Spending Score (1-100)"]])
    inertia_score2.append(kmeans2.inertia_)
plt.plot(range(1, 11), inertia_score2)

centers = pd.DataFrame(clustering2.cluster_centers_)
centers.columns = ["x", "y"]
plt.figure(figsize = (10, 8))
plt.scatter(x = centers["x"], y = centers["y"], s = 100, c = "black", marker = "*")
sns.scatterplot(data = df, x = "Annual Income (k$)", y = "Spending Score (1-100)", hue = "Spending and Income Cluster", palette = "tab10")
plt.savefig("clustering_bivariate.png")
#* -->Bu blok, "Annual Income (k$)" ve "Spending Score (1-100)" sütunları için oluşturulan küme merkezlerini (centers) gösteren bir nokta grafiği oluşturur. Ayrıca, verileri renklendirerek kümeleme sonuçlarını görselleştirir ve grafiği "clustering_bivariate.png" olarak kaydeder.

pd.crosstab(df["Spending and Income Cluster"], df["Gender"], normalize = "index") #* -->"Spending and Income Cluster" ve "Gender" sütunları arasındaki ilişkiyi gösteren bir çapraz tablo oluşturur ve normalize eder.

df.groupby("Spending and Income Cluster")[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].mean()

#Multivariate Clustering

from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
dff = pd.get_dummies(df, drop_first = True)
dff = dff[["Age", "Annual Income (k$)", "Spending Score (1-100)", "Gender_Male"]]
dff = pd.DataFrame(scale.fit_transform(dff))
#* -->Bu blok, verileri standartlaştırmak için StandardScaler'ı kullanır. Öncelikle, DataFrame'i get_dummies işlemiyle kategorik değişkenleri ikili değişkenlere dönüştürür. Ardından, sadece belirli sütunları seçer ve standartlaştırma işlemi uygular.

inertia_score3 = []
for i in range(1, 11):
    kmeans3 = KMeans(n_clusters = i)
    kmeans3.fit(dff)
    inertia_score3.append(kmeans3.inertia_)
plt.plot(range(1, 11), inertia_score3)
#* -->Bu blok, standartlaştırılmış veriler için farklı küme sayıları için inertia değerlerini hesaplar ve bu değerleri bir grafik üzerinde gösterir.

df.to_csv("Clustering.csv")