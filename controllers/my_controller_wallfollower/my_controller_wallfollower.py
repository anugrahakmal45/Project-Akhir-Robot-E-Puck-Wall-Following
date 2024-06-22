from controller import Robot
import matplotlib.pyplot as plt
import numpy as np 
import skfuzzy as fuzz

# create the Robot instance.
robot = Robot()
# Mendapatkan nilai timestep dari lingkungan simulasi
timestep = int(robot.getBasicTimeStep())

left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')

# Mengatur motor agar berputar tanpa batas dan memiliki kecepatan awal 0
left_motor.setPosition(float('inf'))    
left_motor.setVelocity(0.0)
right_motor.setPosition(float('inf'))
right_motor.setVelocity(0.0)

# definisi sensor
sensor_ps5 = robot.getDevice('ps5')
sensor_ps6 = robot.getDevice('ps6')
sensor_ps7 = robot.getDevice('ps7')

sensor_ps5.enable(timestep)
sensor_ps6.enable(timestep)
sensor_ps7.enable(timestep)

#definisi array fuzzy
x_sensor = np.arange(0, 101, 1)
x_motor = np.arange(-45, 46, 1)

#alat konversi
def map_value(value):
    return int((value + 100) * 2.5)

def calculate_motor(signal):
    return (signal/10)
 
while robot.step(timestep) != -1:

    # Memproses data sensor
    value_ps5 = sensor_ps5.getValue()
    value_ps6 = sensor_ps6.getValue()
    value_ps7 = sensor_ps7.getValue() 
    if value_ps5 >= 100:
        value_ps5 = 100
    if value_ps6 >= 100:
        value_ps6 = 100
    if value_ps7 >= 100:
        value_ps7 = 100
    sensor_5_map = map_value(-value_ps5)
    sensor_6_map = map_value(-value_ps6)
    sensor_7_map = map_value(-value_ps7)

    #Define Membership Function
    ps5_dekat = fuzz.trapmf(x_sensor, [0, 0, 20, 50])
    ps5_sedang = fuzz.trimf(x_sensor, [ 20, 50, 80])
    ps5_jauh = fuzz.trapmf(x_sensor, [50, 80, 100, 100])
    
    ps6_dekat = fuzz.trapmf(x_sensor, [0, 0, 20, 50])
    ps6_sedang = fuzz.trimf(x_sensor, [20, 50, 80])
    ps6_jauh = fuzz.trapmf(x_sensor, [50, 80, 100, 100])
    
    ps7_dekat = fuzz.trapmf(x_sensor,[0, 0, 20, 50])
    ps7_sedang = fuzz.trimf(x_sensor, [20, 50, 80])
    ps7_jauh = fuzz.trapmf(x_sensor, [50, 80, 100, 100])
    
    mtrkanan_mundur = fuzz.trapmf(x_motor, [-45, -45, 0, 10])
    mtrkanan_pelan = fuzz.trimf(x_motor, [0, 10, 20])
    mtrkanan_normal = fuzz.trimf(x_motor, [10, 20, 30])
    mtrkanan_cepat = fuzz.trimf(x_motor, [20, 35, 45])
    
    mtrkiri_mundur = fuzz.trapmf(x_motor, [-45, -45, 0, 10])
    mtrkiri_pelan = fuzz.trimf(x_motor, [0, 10, 20])
    mtrkiri_normal = fuzz.trimf(x_motor, [10, 20, 30])
    mtrkiri_cepat = fuzz.trimf(x_motor, [20, 35, 45])

    
    #Input Value membership
    ps5_lvl_dekat = fuzz.interp_membership(x_sensor, ps5_dekat, sensor_5_map)
    ps5_lvl_sedang = fuzz.interp_membership(x_sensor, ps5_sedang, sensor_5_map)
    ps5_lvl_jauh = fuzz.interp_membership(x_sensor, ps5_jauh, sensor_5_map)

    ps6_lvl_dekat = fuzz.interp_membership(x_sensor, ps6_dekat, sensor_6_map)
    ps6_lvl_sedang = fuzz.interp_membership(x_sensor, ps6_sedang, sensor_6_map)
    ps6_lvl_jauh = fuzz.interp_membership(x_sensor, ps6_jauh, sensor_6_map)

    ps7_lvl_dekat = fuzz.interp_membership(x_sensor, ps7_dekat, sensor_7_map)
    ps7_lvl_sedang = fuzz.interp_membership(x_sensor, ps7_sedang, sensor_7_map)
    ps7_lvl_jauh = fuzz.interp_membership(x_sensor, ps7_jauh, sensor_7_map)

    #Rule
    rules_1 = np.fmin(ps7_lvl_dekat, np.fmin(ps6_lvl_dekat, ps5_lvl_dekat)) #CEPAT, MUNDUR
    rules_2 = np.fmin(ps7_lvl_sedang, np.fmin(ps6_lvl_dekat, ps5_lvl_dekat)) #CEPAT, LAMBAT
    rules_3 = np.fmin(ps7_lvl_jauh, np.fmin(ps6_lvl_dekat, ps5_lvl_dekat)) #CEPAT,LAMBAT
    rules_4 = np.fmin(ps7_lvl_dekat, np.fmin(ps6_lvl_sedang, ps5_lvl_dekat)) #CEPAT, MUNDUR
    rules_5 = np.fmin(ps7_lvl_sedang, np.fmin(ps6_lvl_sedang, ps5_lvl_dekat)) #CEPAT, NORMAL
    rules_6 = np.fmin(ps7_lvl_jauh, np.fmin(ps6_lvl_sedang, ps5_lvl_dekat)) #NORMAL, LAMBAT
    rules_7 = np.fmin(ps7_lvl_dekat, np.fmin(ps6_lvl_jauh, ps5_lvl_dekat)) #CEPAT, MUNDUR
    rules_8 = np.fmin(ps7_lvl_sedang, np.fmin(ps6_lvl_jauh, ps5_lvl_dekat)) #MUNDUR, CEPAT
    rules_9 = np.fmin(ps7_lvl_jauh, np.fmin(ps6_lvl_jauh, ps5_lvl_dekat)) #MUNDUR, lAMBAT
    rules_10 = np.fmin(ps7_lvl_dekat, np.fmin(ps6_lvl_dekat, ps5_lvl_sedang))  # CEPAT, MUNDUR
    rules_11 = np.fmin(ps7_lvl_sedang, np.fmin(ps6_lvl_dekat, ps5_lvl_sedang)) #NORMAL, NORMAL
    rules_12 = np.fmin(ps7_lvl_jauh, np.fmin(ps6_lvl_dekat, ps5_lvl_sedang)) #LAMBAT, CEPAT
    rules_13 = np.fmin(ps7_lvl_dekat, np.fmin(ps6_lvl_sedang, ps5_lvl_sedang)) #CEPAT, MUNDUR
    rules_14 = np.fmin(ps7_lvl_sedang, np.fmin(ps6_lvl_sedang, ps5_lvl_sedang)) #NORMAL, MUNDUR
    rules_15 = np.fmin(ps7_lvl_jauh, np.fmin(ps6_lvl_sedang, ps5_lvl_sedang)) #MUNDUR, NORMAL
    rules_16 = np.fmin(ps7_lvl_dekat, np.fmin(ps6_lvl_jauh, ps5_lvl_sedang)) #LAMBAT, CEPAT
    rules_17 = np.fmin(ps7_lvl_sedang, np.fmin(ps6_lvl_jauh, ps5_lvl_sedang)) #CEPAT, CEPAT
    rules_18 = np.fmin(ps7_lvl_jauh, np.fmin(ps6_lvl_jauh, ps5_lvl_sedang)) #CEPAT, CEPAT
    rules_19 = np.fmin(ps7_lvl_dekat, np.fmin(ps6_lvl_dekat, ps5_lvl_jauh)) #CEPAT, LAMBAT
    rules_20 = np.fmin(ps7_lvl_sedang, np.fmin(ps6_lvl_dekat, ps5_lvl_jauh)) #CEPAT, MUNDUR
    rules_21 = np.fmin(ps7_lvl_jauh, np.fmin(ps6_lvl_dekat, ps5_lvl_jauh)) #LAMBAT, CEPAT
    rules_22 = np.fmin(ps7_lvl_dekat, np.fmin(ps6_lvl_sedang, ps5_lvl_jauh)) #NORMAL, CEPAT
    rules_23 = np.fmin(ps7_lvl_sedang, np.fmin(ps6_lvl_sedang, ps5_lvl_jauh)) #CEPAT, LAMBAT
    rules_24 = np.fmin(ps7_lvl_jauh, np.fmin(ps6_lvl_sedang, ps5_lvl_jauh)) #CEPAT, CEPAT
    rules_25 = np.fmin(ps7_lvl_dekat, np.fmin(ps6_lvl_jauh, ps5_lvl_jauh)) #MUNDUR, CEPAT
    rules_26 = np.fmin(ps7_lvl_sedang, np.fmin(ps6_lvl_jauh, ps5_lvl_jauh)) #CEPAT, NORMAL
    rules_27 = np.fmin(ps7_lvl_jauh, np.fmin(ps6_lvl_jauh, ps5_lvl_jauh)) #CEPAT, CEPAT
    
    motor0 = np.zeros_like(x_motor)
    
    #Ouput Motor Kanan
    pelan_kanan = np.fmin(np.fmax(rules_2, np.fmax(rules_3,np.fmax(rules_6,np.fmax(rules_21, np.fmax(rules_19, rules_23))))), mtrkanan_pelan)
    normal_kanan = np.fmin(np.fmax(rules_8,np.fmax(rules_9,np.fmax(rules_5,np.fmax(rules_7,np.fmax(rules_11, rules_26))))), mtrkanan_normal)
    cepat_kanan = np.fmin(np.fmax(rules_12,np.fmax(rules_16,np.fmax(rules_17,np.fmax(rules_18,np.fmax(rules_19,np.fmax(rules_22, np.fmax(rules_24,np.fmax(rules_25, rules_27)))))))), mtrkanan_cepat)
    mundur_kanan = np.fmin(np.fmax(rules_1,np.fmax(rules_4,np.fmax(rules_10,np.fmax(rules_13, np.fmax(rules_14,rules_20))))), mtrkanan_mundur)
    
    #Output_Motor Kiri
    pelan_kiri = np.fmin(np.fmax(rules_8,np.fmax(rules_9,np.fmax(rules_12,rules_16))), mtrkiri_pelan)
    normal_kiri = np.fmin(np.fmax(rules_6,np.fmax(rules_21, np.fmax(rules_11, np.fmax(rules_14, rules_22)))), mtrkiri_normal)
    cepat_kiri = np.fmin(np.fmax(rules_1, np.fmax(rules_3, np.fmax(rules_2, np.fmax(rules_4, np.fmax(rules_5, np.fmax(rules_7, np.fmax(rules_8, np.fmax(rules_10,np.fmax(rules_13,  np.fmax(rules_17,np.fmax(rules_18, np.fmax(rules_19, np.fmax(rules_20, np.fmax(rules_23, np.fmax(rules_24, rules_26))))))))))))))), mtrkiri_cepat)
    mundur_kiri = np.fmin(np.fmax(rules_15,np.fmax(rules_26,rules_27)), mtrkiri_mundur)
    
    #Aggregate all three output membership function together
    aggregated_kiri = np.fmax(np.fmax(pelan_kiri, normal_kiri), np.fmax(cepat_kiri, mundur_kiri))
    kecepatan_motor_kiri  = fuzz.defuzz(x_motor, aggregated_kiri, 'centroid')
    hasil_motor_kiri = fuzz.interp_membership(x_motor, aggregated_kiri, kecepatan_motor_kiri)
    
    #Aggregate all three output membership function together
    aggregated_kanan = np.fmax(np.fmax(pelan_kanan, normal_kanan), np.fmax(cepat_kanan, mundur_kanan))
    kecepatan_motor_kanan  = fuzz.defuzz(x_motor, aggregated_kanan, 'centroid')
    hasil_motor_kanan = fuzz.interp_membership(x_motor, aggregated_kanan, kecepatan_motor_kanan)
       
    kecepatan_motor_kiri_fix = calculate_motor(kecepatan_motor_kiri)
    kecepatan_motor_kanan_fix = calculate_motor(kecepatan_motor_kanan)
    print(f"sensor 5 : {value_ps5} || sensor 6: {value_ps6} || sensor 7: {value_ps7} || map5:  {sensor_5_map} || map6: {sensor_6_map} || map7: {sensor_7_map} || motor kanan: {kecepatan_motor_kanan_fix} || motor kiri: {kecepatan_motor_kiri_fix}")

    left_motor.setVelocity(kecepatan_motor_kiri_fix)
    right_motor.setVelocity(kecepatan_motor_kanan_fix)
    

    