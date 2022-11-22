import os
 
def Rename(folder, ekstensi):
    for num, filename in enumerate(os.listdir(folder)):

        src =f"{folder}/{filename}"
        dst =f"{folder}/{filename.split('.')[0] + '.' + ekstensi}"
        os.rename(src, dst)

def main():
   
    nama = input("Masukkan nama folder: ")
    parent = input("Masukkan folder parent ('Dataset' atau 'Datatest') : ")
    ekstensi = input("Masukkan ekstensi (contoh : 'jpg' ): ")
    folder = "./test/" + parent + "/" + nama

    Rename(folder, ekstensi)
 
if __name__ == '__main__':
     
    main()