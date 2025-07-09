import matplotlib.pyplot as plt
import pandas as pd

hujan = pd.read_csv("dataset/data_hujan.csv")
suhu = pd.read_csv("dataset/data_suhu.csv")
kelembapan = pd.read_csv("dataset/data_kelembapan.csv")

hujan_2006 = [165.11813731, 196.02367945, 185.76474245, 142.3923215, 115.43254657, 158.18893444, 133.84723559, 243.93796515, 282.94276769, 225.62192719, 99.20007074, 217.14192977]
suhu_2006 = [30.88225227, 32.32980718, 32.75674966, 32.78643455, 32.94301279, 32.83837429, 32.0428363, 32.71928074, 31.79822307, 31.99417408, 31.56290521, 30.51107463]
kelembapan_2006 = [71.39204111, 71.11383407, 62.69075531, 65.79497095, 59.02282968, 64.45010214, 66.01676806, 70.06870083, 65.88965779, 64.91817797, 65.92283284, 64.96403981]

fig, axs = plt.subplots(3)

axs[0].plot(hujan["1997"], color="Black", label="1997", linewidth=1)
axs[0].plot(hujan["1998"], color="Brown", label="1998", linewidth=1)
axs[0].plot(hujan["1999"], color="Red", label="1999", linewidth=1)
axs[0].plot(hujan["2000"], color="Orange", label="2000", linewidth=1)
axs[0].plot(hujan["2001"], color="Yellow", label="2001", linewidth=1)
axs[0].plot(hujan["2002"], color="Green", label="2002", linewidth=1)
axs[0].plot(hujan["2003"], color="Blue", label="2003", linewidth=1)
axs[0].plot(hujan["2004"], color="Purple", label="2004", linewidth=1)
axs[0].plot(hujan["2005"], color="Grey", label="2005", linewidth=1)
axs[0].plot(hujan_2006, color="Black", label="2006", linestyle="dashdot", linewidth=3)

axs[1].plot(suhu["1997"], color="Black", label="1997", linewidth=1)
axs[1].plot(suhu["1998"], color="Brown", label="1998", linewidth=1)
axs[1].plot(suhu["1999"], color="Red", label="1999", linewidth=1)
axs[1].plot(suhu["2000"], color="Orange", label="2000", linewidth=1)
axs[1].plot(suhu["2001"], color="Yellow", label="2001", linewidth=1)
axs[1].plot(suhu["2002"], color="Green", label="2002", linewidth=1)
axs[1].plot(suhu["2003"], color="Blue", label="2003", linewidth=1)
axs[1].plot(suhu["2004"], color="Purple", label="2004", linewidth=1)
axs[1].plot(suhu["2005"], color="Grey", label="2005", linewidth=1)
axs[1].plot(suhu_2006, color="Black", label="2006", linestyle="dashdot", linewidth=3)

axs[2].plot(kelembapan["1997"], color="Black", label="1997", linewidth=1)
axs[2].plot(kelembapan["1998"], color="Brown", label="1998", linewidth=1)
axs[2].plot(kelembapan["1999"], color="Red", label="1999", linewidth=1)
axs[2].plot(kelembapan["2000"], color="Orange", label="2000", linewidth=1)
axs[2].plot(kelembapan["2001"], color="Yellow", label="2001", linewidth=1)
axs[2].plot(kelembapan["2002"], color="Green", label="2002", linewidth=1)
axs[2].plot(kelembapan["2003"], color="Blue", label="2003", linewidth=1)
axs[2].plot(kelembapan["2004"], color="Purple", label="2004", linewidth=1)
axs[2].plot(kelembapan["2005"], color="Grey", label="2005", linewidth=1)
axs[2].plot(kelembapan_2006, color="Black", label="2006", linestyle="dashdot", linewidth=3)

plt.show()