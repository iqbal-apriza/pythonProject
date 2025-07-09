import numpy as np

# my_age = 15
# my_resp_rate = 10

my_temp = 35
my_heart_beat = 150

pelan = 150
biasa = 200
cepat = 250

sedikit = 20
normal = 30
banyak = 40

rule = np.zeros([9])


def bahu_kiri(mini, maxi, val):
    if val < mini:
        return 1
    elif val >= mini and val <= maxi:
        return (maxi - val) / (maxi - mini)
    elif val > maxi:
        return 0

def bahu_kanan(mini, maxi, val):
    if val < mini:
        return 0
    elif val >= mini and val <= maxi:
        return (val - mini) / (maxi - mini)
    elif val > maxi:
        return 1

def segitiga(mini, mid, maxi, val):
    if val < mini or val > maxi:
        return 0
    elif val >= mini and val < mid:
        return (val - mini) / (mid - mini)
    elif val > mid and val <= maxi:
        return (maxi - val) / (maxi - mid)
    elif val == mid:
        return 1

def trapesium(mini, left_val, right_val, maxi, val):
    if val < mini or val > maxi:
        return 0
    elif val >= mini and val <= left_val:
        return (val - mini) / (left_val - mini)
    elif val > left_val and val < right_val:
        return 1
    elif val >= right_val and val <= maxi:
        return (maxi - val) / (maxi - right_val)


# umur_muda = bahu_kiri(7, 17, my_age)
# umur_dewasa = trapesium(7, 17, 50, 60, my_age)
# umur_tua = bahu_kanan(50, 60, my_age)

detak_lambat = bahu_kiri(50, 60, my_heart_beat)
detak_normal = trapesium(50, 60, 100, 110, my_heart_beat)
detak_cepat = bahu_kanan(100, 110, my_heart_beat)

# slow_resp_rate = bahu_kiri(10, 12, my_resp_rate)
# normal_resp_rate = trapesium(10, 12, 20, 22, my_resp_rate)
# high_resp_rate = bahu_kanan(20, 22, my_resp_rate)

suhu_rendah = bahu_kiri(35, 36, my_temp)
suhu_normal = trapesium(35, 36, 37, 38, my_temp)
suhu_tinggi = bahu_kanan(37, 38, my_temp)

rule[0] = min(suhu_rendah, detak_lambat)
rule[1] = min(suhu_rendah, detak_normal)
rule[2] = min(suhu_rendah, detak_cepat)
rule[3] = min(suhu_normal, detak_lambat)
rule[4] = min(suhu_normal, detak_normal)
rule[5] = min(suhu_normal, detak_cepat)
rule[6] = min(suhu_tinggi, detak_lambat)
rule[7] = min(suhu_tinggi, detak_normal)
rule[8] = min(suhu_tinggi, detak_cepat)

defuzz_speed = ((pelan * rule[0]) + (pelan * rule[1]) + (pelan * rule[2]) + (biasa * rule[3]) + (biasa * rule[4]) + (biasa * rule[5]) + (cepat * rule[6]) + (cepat * rule[7]) + (cepat * rule[8])) / sum(rule)

# print(umur_muda, umur_dewasa, umur_tua)
print(detak_lambat, detak_normal, detak_cepat)
# print(slow_resp_rate, normal_resp_rate, high_resp_rate)
print(suhu_rendah, suhu_normal, suhu_tinggi)
print(rule)
print(defuzz_speed)