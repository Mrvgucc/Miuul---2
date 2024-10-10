# FEATURE & DATA PRE-PROCESSING
# - Outliers (Aykırı Değerler)
# - Missing Values (Eksik Değerler)
# - Encoding
# - Feature Scaling (Ozellik Olceklendirme)
# - Feature Extraction (Ozellik Cikarimi)
# - Feature Interactions (Ozellik Etkilesimleri)
# - End to End Application (Uctan Uca Uygulama)
# Ozellik Muhendisligi : Ozellikler uzerinden gercklestirilen calismalar. Ham veriden degisken uretmek
# Veri On Isleme : Calismalar oncesi verinin uygun hale getirilmesi surecidir.

# ----------------------------------------------------------------------------------------------------------------------

# OUTLIERS (Aykiri Değerler)
# Verideki genel eğilimin oldukça disina cikan degerlere aykiri degerler denir.
# Aykiri deger neye sebep olur?
# Ozellikle dogrusal degerlerde aykiri degerlerin etkileri daha siddetlidir.
# Agac yontemlerinde;bu etkiler daha dusuktur.
# Aykırı degerler neye gor belirlenir ?
# 1. Sektor Bilgisi
# 2. Standart Sapma Yaklasimi: Degiskenin ortalamasi ver standart sapmasi alinir, bu iki degerin uzerindeki ve altindaki degerler aykiri deger olarak nitelendirilir.
# 3. Z-Skoru Yaklasimi: ilgili degisken standart normal dagilimma uyarlanir.
# 4. Boxplot (interquartile range - IQR) Yontemi: burada da bu yontem kullanilacaktir.
# IQR Yontemi: Q1 -> 25 lik ceyrek; Q3 -> 75 lik ceyrek olsun. Bu ceyrek degerlerine bakilarak, alt ve ust sinir limitleri belirlenir
# Alt sinir: Q1 - 1.5 * IQR  ---  Ust sinir: Q3 + 1.5 * IQR ile hesaplanir. [IQR = Q3 - Q1]
#NOT: eger degiskienimizdeki tum degerler pozitif ise alt sinir calismiyor olacaktir. Bu sebeple genelde ust sinira gore calisiliyor.
# Interquartile Range = IQR

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None) # Butun sutunlari gosterir.
pd.set_option('display.max_rows', None) # Butun satirlari gosterir
pd.set_option('display.float_format', lambda x: '%.3f' % x) # virgulden sonra yalnizca 3 basamak gosterir
pd.set_option('display.width', 500) # ciktinin kaydirilmadan daha duzgun gorunmesini saglar.


# Buyuk olcekli ornekler icin "application_train" dataseti kullanilicaktir.
def load_application_train():
    data = pd.read_csv(r"C:/Users/merve/OneDrive/Masaüstü/datasets/application_train.csv")
    return data

df = load_application_train()
df.head()


# Kucuk olcekli ornekler icin "titanic" veri seti kullanilacaktir.
def laod():
    data = pd.read_csv(r"C:/Users/merve/OneDrive/Masaüstü/datasets/titanic.csv")
    return data

df = laod()
df.head()



# ----------------------------------------------------------------------------------------------------------------------

# Aykiri Degerleri Yakalama

# Grafik teknik ile aykiri degerleri gormek istersek kutu grafik kullanilir
sns.boxplot(x=df["Age"]) # Age degiskeninin dagilim bilgisini verir.
plt.show()

# Kutu grafiginde gozlemlenen aykiri degerlere nasil erisilebilir/ Aykiri degerler nasil yakalani ?
# Oncelikle ceyrek degerlerinin hesaplanmasi gerekir

q1 = df["Age"].quantile(0.25)
q3 = df["Age"].quantile(0.75)
iqr = q3 -q1
up = q3 + (1.5 * iqr)
low = q1 - (1.5 * iqr) # Age degiskeninde - degerler olmadigindan yas alt sinir degeri negatif geldi
df[(df["Age"]<low) | (df["Age"] > up)] # alt sinirdan kucuk olanlari, ust sinirdan buyuk olan degerleri cekelim
df[(df["Age"]<low) | (df["Age"] > up)].index # bu aykiri degerlerin indexlerini elde ederiz.

# Sadece hizli bir sekilde aykiri deger var mi yok mu diye kontrol etmek istersek;
df[(df["Age"]<low) | (df["Age"] > up)].any(axis=None) # satir ya da sutun olarak kontrol etmesin hepsini kontrol etsin diye axis=None yaptik
df[~((df["Age"]<low) | (df["Age"] > up))].any(axis=None) # aykiri olmayan deger var mi yok mu yu ceker (true ya da false olarak)

# Bu kisimda:
# 1. Esik deger belirledik.
# 2. Aykirilara eristik.
# 3. Hizlica aykiri deger var mi yok mu diye sorduk

# ----------------------------------------------------------------------------------------------------------------------

# ISLEMLERI FONKSIYONLASTIRMA
# Aykiri deger yakalama islemini fonksiyonlastiracagiz.

# bu fonk ile her seferinde q1, q3, iqr hesabi ile ugrasmayacagim
def outlier_thresholds(dataFrame, col_name, q1=0.25, q3=0.75): # Aykiri deger icin esik degerleri hesaplayan fonk.
    quartile1 = dataFrame[col_name].quantile(q1)
    quartile3 = dataFrame[col_name].quantile(q3)
    iqr = quartile3 - quartile1
    up_limit = quartile3 + (1.5 * iqr)
    low_limit = quartile1 - (1.5* iqr)
    return low_limit, up_limit

outlier_thresholds(df,"Age")
outlier_thresholds(df,"Fare")

low, up = outlier_thresholds(df,"Fare")

# Aykiri deger var mi yok mu kontrolu yapan fonk (1. yol):
def my_check_outlier(dataFrame, col_name, low, up):
    if dataFrame[dataFrame([col_name] < low) | dataFrame([col_name] > up)].any(axis=None): # tum satir ve sutunlara gore kiyaslanmasini istedigimiz icin axis=None tanimladik
        return True
    else:
        return False
# Aykiri deger var mi yok mu kontrolu yapan fonk (2. yol):
# second way better than first way ***
def check_outlier(dataFrame, col_name):
    low_limit, up_limit = outlier_thresholds(dataFrame,col_name)
    if dataFrame[(dataFrame[col_name] > up_limit) | (dataFrame[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df,"Age")
check_outlier(df,"Fare")

# Burada da soyle bir problem cikiyor: veri seti icinde onlarca degisken oldugunu dusunelim;
# Boyle bir senaryoda her degisken icin tek tek check_outlier() fonksiyonunu calistirmak mantikli degildir.
# Bunun icin de bir fonksiyon olusturmaliyiz.

# ----------------------------------------------------------------------------------------------------------------------

# ISLEMLERI FONKSIYONLASTIRMA - II
# Veri setindeki butun degisken icerikleri sayisal olmaz. Bu durumda yazdigimiz fonksiyonun degisken iceriginin sayisal
# olup olmadigini anlayip ona gore islem yapmasi gerekecektir. (Belirli sayisal degiskenlerin secilmesi gerekir.)

dff = load_application_train() # yukledigimiz buyuk olan veri seti
dff.head()

# Oyle bir islem yapmamiz gerekir ki otomatik olarak;
# - sayisal degiskenleri
# - kategorik degiskenleri
# - kategorik olmasa bile aslinda kategorik olan degiskenleri
# - kategorik oldugu halde aslinda kategorik olmayan degiskenleri
# getirmis olsun.

# Kategorik degisken: ornegin cinsiyet kategorik degiskendir, yani kadin veya erkek olarak nitelendirilebilir.
# Sayisal gorunumlu kategorik degisken: ornegin Survived degiskeni  hayatta kalip kalmama durumunu degerlendiren bir degiskendir
# ve bu degisken 0 ve 1 degerlerini alir ancak aslinda kategorik degiskendir 0 -> hayatta kalmamis; 1 -> hayatta kalmis olarak degerlendirilir.
# Veri setlerinde sayisal tipte oldugu halde kategorik olan bazi degiskenler bulunabilir ve bunlarin yakalanmasi analiz edilmesi gerekebilir.
# Kategorik gorunmlu olup bilgi tasimayan, seyrekligi cok fazla olan, yani cok fazla sinifa sahip olan degiskenler degiskenler:
#Bunlara kardinalligi yuksek olan degiskenler de denir.
# Bir degiskenin kategorik olabailmesi icin olcum degeri tasimalidir

# Bu fonksiyon ile butun bu degiskenl tiplerinin ayrimi yapilabilecek
def grap_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kordinal degiskenlerin isimlerini verir.
    Not: Kategorik degiskenlerin içerisinde numerik gorunumlu kategorik degiskenler de dahildir.

    Parameters
    -----
        dataframe: dataframe
                Degisken isimleri alinmak istenilen dataframe
        cat_th: int, optinal
                numerik fakat kategorik olan degiskenler icin sinif esik degeri
        car_th: int, optinal
                kategorik fakat kordinal degiskenler icin sinif esik degeri

    Returns
    -----
        cat_cols: list
                Kategorik degiskenlerin listesi
        num_cols: list
                Numerik degiskenlerin listesi

    """
    # Kategorik degiskenleri secelim:
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"] # data tipi object ise kategorik olarak siniflandirir.

    # Numerik ama kategorik olan degiskenlerin secimi:
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique()<cat_th and dataframe[col].dtypes != "O"]
    # cat_th = bir degisken 10 dan az bir sinifa sahipse sayisal olsa bile bu degisken benim icin kategoriktir.

    # Kategorik ama cardinal degiskenler:
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    # car_th = eger bir kategorik degisken 20 den fazla sinifi varsa

    # cat_cols listemizi bastan olusturalim:
    cat_cols = cat_cols + num_but_cat
    # cat_cols listesine bir guncelleme daha yapmamiz lazim. Bu cat_cols listesinde kardinalitesi yuksek olan degiskenleri de elemeliyiz
    cat_cols = [col for col in cat_cols if col not in cat_but_car] # kardinal ama kategorik olmayanlari secer

    # numerik colums lar:
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    # numerik gorunumlu kategorik olanlarin cikarilmasi gerekir bu sebeple
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grap_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

for col in num_cols:
    print(col,check_outlier(df,col)) # num_cols ta aykiri degerler kontrol edilir


cat_cols, num_cols, cat_but_car = grap_col_names(dff)

for col in num_cols:
    print(col,check_outlier(dff,col))


# ----------------------------------------------------------------------------------------------------------------------

# AYKIRI DEGERLERE ERISMEK
