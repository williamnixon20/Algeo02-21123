import os
 
def main():
   
    nomor = "3"
    test = True
    folder = ""
    if (test):
        folder = "./test/Datatest/Datatest" + nomor

    else:

        folder = "./test/Dataset/Dataset" + nomor

    for num, filename in enumerate(os.listdir(folder)):
        dst = filename.replace(".", "") + ".jpg"
        src =f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
        dst =f"{folder}/{dst}"
         
        # rename() function will
        # rename all the files
        os.rename(src, dst)
 
# Driver Code
if __name__ == '__main__':
     
    # Calling main() function
    main()