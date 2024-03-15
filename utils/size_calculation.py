from PIL import Image
import pandas as pd

data=pd.read_csv('data/data.csv')
photo_size=[]
seq_length=[]
for i in range(833):
    caption=data.iloc[i]['caption'].strip()
    caption=caption.split()
    seq_length.append(len(caption))
    image=Image.open(f'data/images/{i}.png')
    photo_size.append(image.size)


min_size_photo=min(photo_size)
max_seq_length=max(seq_length)

if __name__=="__main__":
    print(min_size_photo)
    print(max_seq_length)
