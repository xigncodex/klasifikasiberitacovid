from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
from fractions import Fraction
import json
from datetime import datetime
import os
import shutil
import math
import pandas as pd
print("Program start at = ", datetime.now().time())

factory = StemmerFactory()
stemmer = factory.create_stemmer()


#Kode program untuk menghilangkan tanda baca
def preprocessing(berita):
    subs2 = '\d|\W'
    cleaning_words = re.sub(subs2, " ", berita)
    clean = cleaning_words.lower()
    tokenize = clean.split()
    token = " ".join(tokenize)
    return stemmer.stem(token)

#Berfungsi untuk membagi membuat set sebanyak fold yang di input
def makeSubset(k):
    with open('data/data.json') as data:
        doc_berita = json.load(data)

    if os.path.exists('postData/duplikat.txt'):
        os.remove('postData/duplikat.txt')

    dupli = open('postData/duplikat.txt', "a")

    list_judul = []

    print(f'Jumlah Data: {len(doc_berita)}')

    for berita in doc_berita:
        judul_filter = berita['judul']
        if judul_filter not in list_judul:
            list_judul.append(berita['judul'])
        else:
            dupli.write(berita['judul'])
            del berita['judul']
            del berita['cat']
            del berita['berita']


    #Membuat subset
    subset=[]
    length = math.floor(len(doc_berita) / k)
    conter = 0
    print(f'Jumlah data uji:{length}')
    set=[]
    try:
        for berita in doc_berita:
            temp={}
            conter+=1
            temp['judul'] = preprocessing(berita['judul'])
            temp['cat'] = preprocessing(berita['cat'])
            temp['berita'] = preprocessing(berita['berita'])
            subset.append(temp)
            if conter == length:
                set.append(subset)
                temp={}
                subset=[]
                conter=0
    except:
        print('just checking')

    if os.path.exists(f'postData/set-{k}.json'):
            os.remove(f'postData/set-{k}.json')
    with open(f'postData/set-{k}.json', 'w') as uji:
        json.dump(set,uji)


    with open(f'postData/set-{k}.json') as data:
        data = json.load(data)

    # print(len(data))

    #Membuat k-fold
    for x in range(k):
        data_latih = []
        data_uji = data[x]
        for z in range(k):
            if z != x:
                data_latih+=data[z]
        data_hoax=0
        data_valid=0
        for item in data_latih:
            if item['cat'] == 'hoax':
                data_hoax+=1
            if item['cat'] == 'valid':
                data_valid+=1

        prior = {'hoax': str(Fraction(data_hoax / len(data_latih)).limit_denominator()),
                 'valid': str(Fraction(data_valid / len(data_latih)).limit_denominator())}

        if os.path.exists(f'postData/fold-{x+1}'):
            shutil.rmtree(f'postData/fold-{x+1}')
        os.mkdir(f'postData/fold-{x+1}')
        if os.path.exists(f'postData/fold-{x+1}/data_uji.json'):
            os.remove(f'postData/fold-{x+1}/data_uji.json')
        with open(f'postData/fold-{x+1}/data_uji.json', 'w') as uji:
            json.dump(data_uji,uji)

        if os.path.exists(f'postData/fold-{x+1}/data_latih.json'):
            os.remove(f'postData/fold-{x+1}/data_latih.json')
        with open(f'postData/fold-{x+1}/data_latih.json', 'w') as latih:
            json.dump(data_latih,latih)

        if os.path.exists(f'postData/fold-{x+1}/prior.json'):
            os.remove(f'postData/fold-{x+1}/prior.json')
        with open(f'postData/fold-{x+1}/prior.json', 'w') as prio:
            json.dump(prior,prio)

#Berfungsi untuk membagi data latih berdasarkan kategori
def berita_terkategori(x):
    with open(f'postData/fold-{x + 1}/data_latih.json') as f:
        data = json.load(f)
    berita = {}
    berita_hoax=[]
    berita_valid=[]
    for item in data:
        if item['cat']=='hoax':
            berita_hoax.append(item['berita'])
        if item['cat']=='valid':
            berita_valid.append(item['berita'])
    berita['hoax']=berita_hoax
    berita['valid']=berita_valid

    if os.path.exists(f'postData/fold-{x + 1}/berita_cat.json'):
        os.remove(f'postData/fold-{x + 1}/berita_cat.json')
    with open(f'postData/fold-{x + 1}/berita_cat.json', 'w') as file:
        json.dump(berita, file)

#Berfungsi untuk menentukan fitur berdasarkan kata unik
def termUnik(x):
    with open("data/stopword2016.txt", "r") as stopword:
        stopword = stopword.read().splitlines()
    with open(f'postData/fold-{x + 1}/berita_cat.json') as files:
        data_berita_cat = json.load(files)
    cat = ['hoax', 'valid']
    all_berita = []
    unique = []
    for item in cat:
        data = " ".join(data_berita_cat[item])
        all_berita.append(data)
    kata = " ".join(all_berita)
    token = kata.split(' ')
    for item in token:
        if item:
            if item not in stopword:
                if item not in unique:
                    unique.append(item)
    unique.sort()
    if os.path.exists(f'postData/fold-{x + 1}/term_unik.json'):
        os.remove(f'postData/fold-{x + 1}/term_unik.json')
    with open(f'postData/fold-{x + 1}/term_unik.json', 'w') as file:
        json.dump(unique, file)

#Berfungsi untuk memberikan bobot di semua dokumen data latih
def weighted_berita(x):
    with open(f'postData/fold-{x + 1}/berita_cat.json') as files:
        data_berita_cat = json.load(files)
    categori_tokenize={}
    cat = ['hoax', 'valid']
    for item in cat:
        data = " ".join(data_berita_cat[item])
        data = data.split(' ')
        categori_tokenize[item] = data

    # print(categori_tokenize)
    with open(f'postData/fold-{x + 1}/term_unik.json') as filess:
        term_unik = json.load(filess)
    weight_cat_dict ={}
    for key in categori_tokenize:
        # print(key)
        waight_temp = []
        tes = {}
        for i in range(len(term_unik)):
            score = 0
            for item in categori_tokenize[key]:
                if term_unik[i] == item:
                    score += 1
            if score ==0:
                tes[term_unik[i]] = score
            else:tes[term_unik[i]] =score
            waight_temp.append(score)
        weight_cat_dict[key] = tes

    if os.path.exists(f'hasil/weighted_berita{x}.xlsx'):
        os.remove(f'hasil/weighted_berita{x}.xlsx')
    pd.DataFrame(weight_cat_dict).to_excel(f'hasil/weighted_berita{x}.xlsx')

    if os.path.exists(f'postData/fold-{x + 1}/weighted_berita.json'):
        os.remove(f'postData/fold-{x + 1}/weighted_berita.json')
    with open(f'postData/fold-{x + 1}/weighted_berita.json', 'w') as file:
        json.dump(weight_cat_dict, file)

#Berfungsi untuk memberikan nilai conditional probability pada setiap fitur di setiap kategori
def conProbability(x):
    with open(f'postData/fold-{x + 1}/weighted_berita.json') as files:
        weight_cat_dict = json.load(files)
    with open(f'postData/fold-{x + 1}/term_unik.json') as files:
        term_unik = json.load(files)

    count_fitur = {}
    for item in weight_cat_dict:
        term_count = 0
        for val in weight_cat_dict[item]:
            term_count+=weight_cat_dict[item][val]
        count_fitur[item]= term_count

    # print(count_fitur)
    conproba = {}
    for key in weight_cat_dict:
        temp = {}
        for value in weight_cat_dict[key]:
            poss_term = weight_cat_dict[key][value]
            p_kata = (poss_term + 1) / (count_fitur[key] + len(term_unik))
            temp[value] = str(Fraction(p_kata).limit_denominator())
        conproba[key] = temp

    if os.path.exists(f'hasil/conditional_probability{x}.xlsx'):
        os.remove(f'hasil/conditional_probability{x}.xlsx')
    pd.DataFrame(conproba).to_excel(f'hasil/conditional_probability{x}.xlsx')

    if os.path.exists(f'postData/fold-{x + 1}/conproba.json'):
        os.remove(f'postData/fold-{x + 1}/conproba.json')
    with open(f'postData/fold-{x + 1}/conproba.json', 'w') as file:
        json.dump(conproba, file)

#Untuk preprocessing data uji
def unclassDatauji(x):
    with open(f'postData/fold-{x + 1}/data_uji.json') as f:
        data_uji = json.load(f)

    unclass_data_uji = []
    for item in data_uji:
        temp={}
        temp[item['judul']] = item['berita'].split()
        unclass_data_uji.append(temp)

    if os.path.exists(f'postData/fold-{x + 1}/unclass_data_uji.json'):
        os.remove(f'postData/fold-{x + 1}/unclass_data_uji.json')
    with open(f'postData/fold-{x + 1}/unclass_data_uji.json', 'w') as file:
        json.dump(unclass_data_uji, file)

#Mendapatkan hasil klasifikasi
def hasilKlasifikasi(x):
    with open(f'postData/fold-{x + 1}/data_latih.json') as s:
        data_latih = json.load(s)

    with open(f'postData/fold-{x + 1}/prior.json') as s:
        prior = json.load(s)

    with open(f'postData/fold-{x + 1}/unclass_data_uji.json') as f:
        unclass_uji = json.load(f)

    with open(f'postData/fold-{x + 1}/term_unik.json') as f:
        term_unik = json.load(f)

    with open(f'postData/fold-{x + 1}/conproba.json') as f:
        conproba = json.load(f)


    unclass_data_uji_token = {}

    for item in unclass_uji:
        for key in item:
            uniq = []
            for val in item[key]:
                if val in term_unik:
                    uniq.append(val)
            unclass_data_uji_token[key]=uniq

    probabilitas={}
    hoa = 0
    vali = 0
    for item in data_latih:
        if item["cat"]=='hoax':
            hoa+=1
            probabilitas[item['cat']]=hoa
        if item["cat"] == 'valid':
            vali+=1
            probabilitas[item['cat']]=vali

    hasil_posterior = []
    another_d=[]
    for judul in unclass_data_uji_token:
        # print(judul)
        for key in conproba:
            # print(len(conproba))
            tes = []
            value = []
            for val in conproba[key]:
                if val in unclass_data_uji_token[judul]:
                    con = Fraction(conproba[key][val])
                    tes.append(con)
            tes.append(Fraction(prior[key]))
            con= math.prod(tes)
            # print(con)
            # print('\n')
            another_d.append([key,con])
        hasil_posterior.append(another_d[:len(conproba)])
        for i in range(len(conproba)):
            another_d.pop()
    # print(hasil_posterior)
    j=0
    posterior_judul={}
    final_clasifikasi = {}
    for judul in unclass_data_uji_token:
        for i in range(2):
            label=hasil_posterior[j][i][0]
            value=hasil_posterior[j][i][1]
            posterior_judul[label]=value
        if max(posterior_judul.values()) == 0:
            final_clasifikasi[judul]='Overflow'
        else:
            final_clasifikasi[judul]=max(posterior_judul,key=posterior_judul.get)

        j+=1
    # print(posterior_judul)
    return final_clasifikasi

#Test hasil terhadap data uji
def testing(x):
    with open(f'postData/fold-{x + 1}/data_latih.json') as lat:
        data_latih = json.load(lat)
    with open(f'postData/fold-{x + 1}/data_uji.json') as f:
        data_uji = json.load(f)
    real_class_uji = {}
    for item in data_uji:
        real_class_uji[item['judul']] = item['cat']

    final_clasifikasi = hasilKlasifikasi(x)


    hasil_csv= {}
    overflow=0
    hasil_salah = 0
    hasil_benar = 0
    counter_item = 1
    for item in final_clasifikasi:
        temp = {}
        temp = {'Kategori Seharusnya':real_class_uji[item],'Hasil Prediksi': final_clasifikasi[item]}
        hasil_csv['doc' + str(counter_item)] = temp
        # print(item)
        # print(f"hasil_klasifikasi: {final_clasifikasi[item]}"
        #       f"\nklasifikasi seharusnya:{real_class_uji[item]}")
        if real_class_uji[item] == final_clasifikasi[item]:
            # print('BENAR')
            hasil_benar+=1
        elif final_clasifikasi[item] == 'Overflow':
            # print('Overflow')
            overflow+=1
        else:
            # print('TIDAK BENAR')
            hasil_salah +=1
        counter_item+=1

    # print('Hasil:\n')
    if os.path.exists(f'hasil/hasil{x}.xlsx'):
        os.remove(f'hasil/hasil{x}.xlsx')
    pd.DataFrame(hasil_csv).T.to_excel(f'hasil/hasil{x}.xlsx')
    # print(pd.DataFrame(hasil_csv).T)

    print(f"\nHasil Klasifikasi dengan total data uji: {overflow+hasil_benar+hasil_salah}:\n"
          f"dan total data latih: {len(data_latih)}\n"
          f"Benar:{hasil_benar}\n"
          f"Salah:{hasil_salah}\n"
          f"Undifined:{overflow}\n")

    TP = 0
    TN = 0
    FP = 0
    FN = 0



    for doc in hasil_csv:
        if hasil_csv[doc]['Kategori Seharusnya'] == 'valid' and hasil_csv[doc]['Hasil Prediksi']=='valid':
            TP+=1
        elif hasil_csv[doc]['Kategori Seharusnya'] == 'hoax' and hasil_csv[doc]['Hasil Prediksi']=='valid':
            FN+=1
        elif hasil_csv[doc]['Kategori Seharusnya'] == 'valid' and hasil_csv[doc]['Hasil Prediksi']=='hoax':
            FP+=1
        elif hasil_csv[doc]['Kategori Seharusnya'] == 'hoax' and hasil_csv[doc]['Hasil Prediksi']=='hoax':
            TN+=1
    try:
        print('valid is TF')
        akurasi_benar = ((TP+TN)/(TP+TN+FP+FN))
        precision = round((TP/(TP+FP))*100,2)
        recall = round((TP/(TP+FN))*100,2)
        f1 = 2*(recall*precision)/(recall+precision)
        print(f"Akurasi: {round(akurasi_benar*100,2)}\n"
              f"Precision:{precision}\n"
              f"Recall:{recall}\n"
              f"F1 - Measure:{f1}")
    except:print('')

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for doc in hasil_csv:
        if hasil_csv[doc]['Kategori Seharusnya'] == 'hoax' and hasil_csv[doc]['Hasil Prediksi']=='hoax':
            TP+=1
        elif hasil_csv[doc]['Kategori Seharusnya'] == 'valid' and hasil_csv[doc]['Hasil Prediksi']=='hoax':
            FN+=1
        elif hasil_csv[doc]['Kategori Seharusnya'] == 'hoax' and hasil_csv[doc]['Hasil Prediksi']=='valid':
            FP+=1
        elif hasil_csv[doc]['Kategori Seharusnya'] == 'valid' and hasil_csv[doc]['Hasil Prediksi']=='valid':
            TN+=1
    try:
        print('hoax is TF')
        akurasi_benar = ((TP+TN)/(TP+TN+FP+FN))
        precision = round((TP/(TP+FP))*100,2)
        recall = round((TP/(TP+FN))*100,2)
        f1 = 2*(recall*precision)/(recall+precision)
        print(f"Akurasi: {round(akurasi_benar*100,2)}\n"
              f"Precision:{precision}\n"
              f"Recall:{recall}\n"
              f"F1 - Measure:{f1}")
    except:print('')
    

#Melakukan pengujian dengan fold yang diinput
# def mainMet(k):
#     makeSubset(k)

#     for z in range(k):
#         berita_terkategori(z)
#         termUnik(z)
#         weighted_berita(z)
#         conProbability(z)

#         unclassDatauji(z)
#         testing(z)
#         print(f'Fold ke {z+1} selesai')

# fold = input('Masukan Pengujian Fold:')
# mainMet(int(fold))

k=5
for z in range(k):
    testing(z)
    print(f'Fold ke {z+1} selesai')

print("Program end at = ", datetime.now().time())