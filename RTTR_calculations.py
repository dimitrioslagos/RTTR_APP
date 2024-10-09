import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def solar_heat_gain_calc(he, dt_range, lat, z1, do, a):
#   ####Compute k_solar in SI units from IEEE Std 738-2012##
    a_k_solar = 1
    b_k_solar = 1.148*10e-4
    c_k_solar = -1.108*10e-8
    k_solar = a_k_solar + he*b_k_solar + c_k_solar*he**2
#   #####Compute hc, solar altitude######
    day_of_year = datetime.datetime.now().timetuple().tm_yday
    delta = 23.46*np.sin((284+day_of_year)/365*2*np.pi)*np.pi/180
    omega = (dt_range.hour-12)*15*np.pi/180
    hc = np.rad2deg(np.arcsin(np.cos(np.deg2rad(lat))
                              * np.cos(delta)*np.cos(omega)+np.sin(np.deg2rad(lat))*np.sin(delta)))
#   #compute solar azimuth#
    xi = np.sin(omega)/(np.sin(np.deg2rad(lat))*np.cos(omega)-np.cos(np.deg2rad(lat))*np.tan(delta))
    c = []
    for i in range(len(xi)):
        if xi[i] >= 0:
            if omega[i] >= 0:
                c.append(180)
            else:
                c.append(0)
        else:
            if omega[i] >= 0:
                c.append(360)
            else:
                c.append(180)
    zc = np.deg2rad(np.array(c) + np.rad2deg(np.arctan(xi)))
#   ###Compute theta#####
    theta = np.arccos(np.cos(np.deg2rad(hc))*np.cos(zc-np.deg2rad(z1)))
#   #Compute Qs,total heat flux density in SI units from IEEE Std 738-2012
    c1 = -42.2391
    c2 = 63.8044
    c3 = -1.9220
    c4 = 3.46921e-2
    c5 = -3.61118e-4
    c6 = 1.94318e-6
    c7 = -4.07608e-9
    qs = c1+c2*hc+c3*(hc**2)+c4*(hc**3)+c5*(hc**4)+c6*(hc**5)+c7*(hc**6)
    qse = k_solar*qs
    #qse = k_solar*980
    #################
    solar_heat_gain = pd.DataFrame((a*qse*np.sin(theta)*do).values, index=dt_range, columns=['solar_heat_gain'])
    #solar_heat_gain = pd.DataFrame(980*do, index=dt_range, columns=['solar_heat_gain'])
    solar_heat_gain[solar_heat_gain <= 0] = 0
    return solar_heat_gain


def convection_calc(wind_incidence_angle, t_a, t_s, do, wind_speed, he):
    t_film = (t_a+t_s)/2
    ro_f = (1.293-1.525e-4*he+6.379e-9*he**2)/(1+0.00367*t_film)
    k_f = 2.424e-2+7.477e-5*t_film-4.407e-9*t_film**2
    m_f = 1.458e-6*(t_film+273)**1.5/(t_film+383.4)
    wind_angle_rad = np.deg2rad(wind_incidence_angle)
    k_angle = 1.194-np.cos(wind_angle_rad)+0.194*np.cos(wind_angle_rad*2)+0.368*np.sin(2*wind_angle_rad)
#   Forced Convection Calculation
    n_re = do*ro_f*wind_speed/m_f
    qc1 = k_angle*(1.01+1.347*n_re**0.52)*k_f*(t_s-t_a)
    qc2 = k_angle * 0.754 * (n_re ** 0.6) * k_f * (t_s - t_a)
    forced_convection = pd.concat([qc1,qc2], axis=1).max(axis=1)
#   Natural Convection Calculation
    natural_convection = 3.645*(ro_f**0.5)*(do**0.75)*(t_s - t_a)**1.25
    return pd.concat([forced_convection,natural_convection], axis=1).max(axis=1)


def radiated_heat_loss_calc(do, e, t_s, t_a):
    return 17.8*do*e*(((t_s+273)/100)**4-((t_a+273)/100)**4)

def compute_RTTR(ts,wind_speed,wind_angle, temperature,t_range,latitude,Rac,diameter):

    solar_heat_gain_series = solar_heat_gain_calc(he=0, dt_range=t_range, lat=latitude, z1=90, do=diameter,a=0.8)
    convection_heat_loss_series = convection_calc(wind_incidence_angle=wind_angle, t_a=temperature,
                                              t_s=ts, do=diameter, wind_speed=wind_speed, he=0)

    radiated_heat_loss_series = radiated_heat_loss_calc(do=diameter, e=0.8, t_s=ts, t_a=temperature)

    R = ((Rac.max()-Rac.min())/(Rac.index.max()-Rac.index.min())*(ts-Rac.index.max())+Rac.max()).loc[0]

    I = ((convection_heat_loss_series+radiated_heat_loss_series[0]-solar_heat_gain_series['solar_heat_gain'])/R)**0.5
    return I

def temperature_calculation(I,lat,t_range,Rac,diameter, wind_angle,wind_speed, temperature):
    solar_heat_gain_series = solar_heat_gain_calc(he=0, dt_range=t_range, lat=lat, z1=90, do=diameter,a=0.8)
    Temperature_estimation = pd.DataFrame(columns=['value'],index=t_range)
    for t in t_range:
        Iest = pd.DataFrame(index =range(int(temperature.loc[t].round())*10,1000,20),columns=['I'])
        for Temp_init in range(int(temperature.loc[t].round())*10,1000,20):
            convection_heat_loss_series = convection_calc(wind_incidence_angle=wind_angle, t_a=temperature,
                                                      t_s=Temp_init/10, do=diameter, wind_speed=wind_speed, he=0)

            radiated_heat_loss_series = radiated_heat_loss_calc(do=diameter, e=0.8, t_s=Temp_init/10, t_a=temperature)
            R = ((Rac.max() - Rac.min()) / (Rac.index.max() - Rac.index.min()) * (Temp_init/10 - Rac.index.max()) + Rac.max()).loc[0]
            #R = max(R,Rac.min().loc[0]*0.001)
            Iest.loc[Temp_init,'I'] = (((convection_heat_loss_series.loc[t]+radiated_heat_loss_series.loc[t]-solar_heat_gain_series.loc[t,'solar_heat_gain'])/R)**0.5)
        Iest.fillna(value=0,inplace=True)
        Ti = pd.DataFrame(Iest.index,index=Iest['I'].values.astype('float'))
        Ti.loc[I.loc[t]] = np.nan
        Ti.sort_index(inplace=True)
        i = np.where(Ti.index==I.loc[t])[0][0]
        alpha =(Ti.loc[Ti.index[i+1],:]-Ti.loc[Ti.index[i-1],:])/(Ti.index[i+1]-Ti.index[i-1])
        beta = Ti.loc[Ti.index[i+1],:]-alpha*Ti.index[i+1]
        Temperature_estimation.loc[t,'value'] = ((beta+alpha*I.loc[t])/10).values[0]
    return Temperature_estimation

# dt = pd.to_datetime(datetime.datetime(year=2023,month=7,day=15,hour=12)).tz_localize('utc').tz_convert('Europe/Athens')
# t_range = pd.date_range(dt, periods=1, freq='1h')
#
# #for ts in range(0,10,1):
# ts1=85
# wind_speed = pd.DataFrame(1*[0.5],index=t_range)
# wind_angle = pd.DataFrame(1*[45],index=t_range)
# temperature = pd.DataFrame(1*[40],index=t_range)
# latitude=20
# Rac = pd.DataFrame([0.3067/1000,(1+0.004*60)*(0.3067)/1000], index=[25, 85])
# Ic = compute_RTTR(ts=ts1,wind_speed=wind_speed,wind_angle=wind_angle,
#                   temperature=temperature,t_range=t_range,latitude=latitude,Rac=Rac,diameter=0.01215)
# print(Ic)
# print(np.sqrt(3)*20*(Ic-448)/1000)
# Istart=100
# T1 = temperature_calculation(Istart,latitude,t_range,Rac,diameter=0.017)
# Istep =7216
#
# qs = solar_heat_gain_calc(he=0, dt_range=t_range, lat=latitude, z1=90, do=0.017,a=0.8).values[0]
# DT = 50
# Tp=[T1.values[0]]
# while abs(DT)>=0.0001 and T1.values[0]<=100:
#     qw = convection_calc(wind_incidence_angle=45, t_a=40,
#                          t_s=T1, do=0.017, wind_speed=1.1, he=0).values[0]
#     qr = radiated_heat_loss_calc(do=0.017, e=0.8, t_s=T1, t_a=40).values[0]
#     R1 = (1 + 0.004 * (T1.values[0] - 20)) * 0.215 / 1000
#     DQ = R1 * Istep ** 2 + qs - qw - qr
#     DT = 0.01 * DQ / 332
#     T1 = T1 + DT
#     Tp.append(T1.values[0])
#     print(T1)

# I=[]
# T=[]
# for i in range(50,500,5):
#     print(i)
#     I.append(i)
#     T.append(temperature_calculation(i,latitude,t_range,Rac,diameter=0.017))
#     #T1=T1+DT
#     #R1 = (1 + 0.004 * (T1 - 20)) * 0.215 / 1000
# plot = pd.DataFrame(T, index=I).plot(title="Temperature_Current")
# plt.show()
#
# T2= temperature_calculation(Istep,latitude,t_range,Rac,diameter=0.017).values[0]
# print(T1,T2)
#


