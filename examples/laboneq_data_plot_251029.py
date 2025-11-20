#%%
from laboneq.simple import load
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter  # smoothing을 위한 모듈

#%% 데이터 로드
base_path = r"/home/yalgaeahn/JSAHN/qubit-experiment/examples/RIP_mode3_fixed(off)_mode1_sweep"

rip_data_control_g=[]
rip_data_control_e=[]

sweep_n = 46
for i in range(sweep_n):
    fname1 = os.path.join(base_path, f"1112_rip_control_g_bf_{5.5127 + 0.002*i:.4f}_amp1.0")
    fname2 = os.path.join(base_path, f"1112_rip_control_e_bf_{5.5127 + 0.002*i:.4f}_amp1.0")
    rip_data_control_g.append(load(fname1))
    rip_data_control_e.append(load(fname2))

#%% g state, e state 정의
control_g_g_point=[]
control_g_e_point=[]
control_e_g_point=[]
control_e_e_point=[]

for i in range(sweep_n):
    control_g_g_point.append(rip_data_control_g[i].data['q0']['cal_trace'].g.data)
    control_g_e_point.append(rip_data_control_g[i].data['q0']['cal_trace'].e.data)
    control_e_g_point.append(rip_data_control_e[i].data['q0']['cal_trace'].g.data)
    control_e_e_point.append(rip_data_control_e[i].data['q0']['cal_trace'].e.data)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# ---- 3. control = g, md ----
axes[0].scatter(np.real(control_g_g_point), np.real(control_g_e_point), color='tab:blue', label='Real')
axes[0].scatter(np.imag(control_g_g_point), np.imag(control_g_e_point), color='tab:orange', label='Imag')
axes[0].set_title('Control = g (MD)')
axes[0].set_xlabel('g point')
axes[0].set_ylabel('e point')
axes[0].grid(True)

# ---- 4. control = e, md ----
axes[1].scatter(np.real(control_e_g_point), np.real(control_e_e_point), color='tab:blue', label='Real')
axes[1].scatter(np.imag(control_e_g_point), np.imag(control_e_e_point), color='tab:orange', label='Imag')
axes[1].set_title('Control = e (MD)')
axes[1].set_xlabel('g point')
axes[1].set_ylabel('e point')
axes[1].grid(True)

# 공통 legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')

plt.tight_layout()
plt.show()

#%% duration(공통), phase데이터 추출
duration = rip_data_control_g[1].data['q0']['result'].axis[0][0]

phase_control_g=[]
phase_control_e=[]

for i in range(sweep_n):
    phase_control_g.append(rip_data_control_g[i].data['q0']['result'].data)
    phase_control_e.append(rip_data_control_e[i].data['q0']['result'].data)


#%%
#%%
def normalize_population(points, g, e):
    """
    points: list or numpy array of complex numbers
    return: array of normalized values in [0,1]
    """
    points = np.array(points, dtype=complex)
    d0 = np.abs(points - g)
    d1 = np.abs(points - e)
    return d0 / (d0 + d1)

# 2D 리스트로 데이터를 받도록 구조 변경
new_phase_control_g=[]
new_phase_control_e=[]

# --- 3. new_phase_control_g_md ---
for i in range(len(phase_control_g)):
    row = [] # <-- 수정
    for k in range(len(phase_control_g[i])):
        normalized_value = normalize_population(phase_control_g[i][k], control_g_g_point[i], control_g_e_point[i])
        row.append(normalized_value) # <-- 수정
    new_phase_control_g.append(row) # <-- 수정

# --- 4. new_phase_control_e_md ---
for i in range(len(phase_control_e)):
    row = [] # <-- 수정
    for k in range(len(phase_control_e[i])):
        normalized_value = normalize_population(phase_control_e[i][k], control_e_g_point[i], control_e_e_point[i])
        row.append(normalized_value) # <-- 수정
    new_phase_control_e.append(row) # <-- 수정

#%%
# --- 1. stack 함수 ---
def stack_2d(data_list):
    arr = [np.ravel(np.array(x)) for x in data_list]
    return np.array(arr)

# --- 2. 데이터 정리 ---
Z_g = stack_2d(new_phase_control_g)
Z_e = stack_2d(new_phase_control_e)

bf = np.array([5.49 + 0.005 * i for i in range(sweep_n)])
t = np.array(duration)

# --- 3. Colormap plot ---
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(wspace=0.25, hspace=0.25)

# (3) control=g
im2 = axes[0].pcolormesh(t, bf, Z_g, shading='auto', cmap='viridis')
axes[0].set_title('Control = |g> ', fontsize=18)
axes[0].set_xlabel('Duration(s)', fontsize=16)
axes[0].set_ylabel('$\omega_d$ (GHz)', fontsize=18)
axes[0].tick_params(axis='both', labelsize=16)
#axes[0].set_ylim([6.487, 6.523])
fig.colorbar(im2, ax=axes[0])

# (4) control=e
im3 = axes[1].pcolormesh(t, bf, Z_e, shading='auto', cmap='viridis')
axes[1].set_title('Control = |e>', fontsize=18)
axes[1].set_xlabel('Duration(s)', fontsize=16)
#axes[1].set_ylabel('Base Frequency (GHz)')
axes[1].tick_params(axis='both', labelsize=16)
#axes[1].set_ylim([6.487, 6.523])
fig.colorbar(im3, ax=axes[1])

#plt.suptitle("Normalized population vs duration and base frequency", fontsize=14)
plt.tight_layout()
plt.show()


#%% 1. 피팅을 위한 라이브러리 및 함수 정의
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import hilbert # (선택적: 위상 추출용)
import matplotlib.pyplot as plt

def cos_func(t, A, f, phi, C):
    """
    피팅을 위한 코사인 함수
    t: 시간 (duration)
    A: 진폭 (Amplitude)
    f: 주파수 (Frequency) - 우리가 찾으려는 핵심 값
    phi: 위상 (Phase offset)
    C: 수직 이동 (Offset)
    """
    return A * np.cos(2 * np.pi * f * t + phi) + C

#%%
#%% 2. 램지(Ramsey) 데이터 피팅 헬퍼 함수

def fit_ramsey(t, y):
    """
    t (duration)와 y (population) 데이터를 받아
    cos_func로 피팅하고 최적의 파라미터 [A, f, phi, C]를 반환합니다.
    """
    t = np.asarray(t)
    y = np.asarray(y)

    if t.size == 0 or y.size == 0:
        print("fit_ramsey: empty input detected. Returning NaNs.")
        return [np.nan] * 4

    if t.size != y.size:
        raise ValueError(f"fit_ramsey: t/y size mismatch ({t.size} vs {y.size}).")

    if t.size < 2:
        print("fit_ramsey: need at least two samples. Returning NaNs.")
        return [np.nan] * 4

    # 1. 초기값(p0) 추측
    A_guess = (np.max(y) - np.min(y)) / 2.0
    C_guess = np.mean(y)
    
    # 2. FFT를 사용한 주파수(f) 추측
    dt = t[1] - t[0]  # duration 스텝 크기
    N = len(t)
    
    # 평균값을 빼고 FFT 수행
    fft_y = np.fft.fft(y - C_guess)
    # 주파수 축 생성
    freqs = np.fft.fftfreq(N, dt)
    
    # DC 성분(freqs[0])을 제외하고 가장 큰 주파수 성분을 찾음
    fft_slice = np.abs(fft_y[1 : N // 2])
    if fft_slice.size == 0:
        f_guess = 0.0
    else:
        idx = np.argmax(fft_slice) + 1
        f_guess = np.abs(freqs[idx]) # 주파수는 양수로 가정
    
    p0 = [A_guess, f_guess, 0, C_guess] # A, f, phi, C
    
    try:
        # 3. Curve fitting
        popt, _ = curve_fit(cos_func, t, y, p0=p0, maxfev=10000)
        
        # 진폭(A)이 음수일 경우, 위상(phi)을 pi만큼 조정하여 A를 양수로 만듦
        if popt[0] < 0:
            popt[0] = -popt[0]
            popt[2] = popt[2] + np.pi
            
        return popt
    except RuntimeError:
        # 피팅 실패 시 NaN 반환
        print(f"Fitting failed. Returning NaNs. Guessed f={f_guess:.4f}")
        return [np.nan] * 4

#%% 4. 데이터 피팅
delta_f = []
fits_g = []
fits_e = []

for i in range(len(bf)):
    y_g = Z_g[i]
    y_e = Z_e[i]
    
    popt_g = fit_ramsey(t, y_g)
    popt_e = fit_ramsey(t, y_e)
    
    fits_g.append(popt_g)
    fits_e.append(popt_e)
    
    # 주파수 차이 (f_g - f_e) 저장 (단위: GHz)
    delta_f.append(popt_g[1] - popt_e[1])

print("--- Fitting Complete ---")

#%%
delta_f = []
fits_g = []
fits_e = []

t_cut = 200e-9  # 300 ns (단위: 초)

for i in range(len(bf)):
    y_g = Z_g[i]
    y_e = Z_e[i]
    
    # --- 300ns 이후 데이터만 사용 ---
    mask = t >= t_cut
    t_fit = t[mask]
    y_g_fit = y_g[mask]
    y_e_fit = y_e[mask]
    
    popt_g = fit_ramsey(t_fit, y_g_fit)
    popt_e = fit_ramsey(t_fit, y_e_fit)
    
    fits_g.append(popt_g)
    fits_e.append(popt_e)
    
    # 주파수 차이 (f_g - f_e) 저장 (단위: GHz)
    delta_f.append(popt_g[1] - popt_e[1])

print("--- Fitting Complete ---")

#%% 4. 데이터 피팅
delta_f = []
fits_g = []
fits_e = []

for i in range(len(bf)):
    y_g = Z_g[i]
    y_e = Z_e[i]

    # --- i에 따라 t_cut 다르게 지정 ---
    if i < 5:
        t_cut = 400e-9  # 200 ns
    elif i < 8:
        t_cut = 400e-9  # 200 ns
    elif i < 12:
        t_cut = 400e-9  # 200 ns
    elif i < 13:
        t_cut = 400e-9  # 200 ns
    elif i < 16:
        t_cut = 600e-9  # 300 ns
    elif i < 19:
        t_cut = 600e-9  # 300 ns
    elif i < 22:
        t_cut = 600e-9  # 300 ns
    elif i < 27:
        t_cut = 600e-9  # 300 ns
    elif i < 28:
        t_cut = 600e-9  # 300 ns
    elif i < 28:
        t_cut = 600e-9  # 300 ns
    else:
        t_cut = 400e-9  # 400 ns

    # --- t_cut 이후 데이터만 사용 ---
    mask = t >= t_cut
    t_fit = t[mask]
    y_g_fit = y_g[mask]
    y_e_fit = y_e[mask]

    popt_g = fit_ramsey(t_fit, y_g_fit)
    popt_e = fit_ramsey(t_fit, y_e_fit)

    fits_g.append(popt_g)
    fits_e.append(popt_e)

    delta_f.append(popt_g[1] - popt_e[1])

print("--- Fitting Complete ---")

#%% 5. [결과 1] Delta_f (ZZ coupling) vs. Base Frequency 플롯

# GHz -> MHz 변환
delta_f_mhz = np.array(delta_f)

fig= plt.subplots(figsize=(5, 6), sharey=True)
#fig.suptitle("Conditional Frequency Shift ($\Delta f = f_g - f_e$) vs. Base Frequency", fontsize=16, y=1.02)

# (2) MD Plot
plt.plot(delta_f_mhz, bf, 'o', color='red', linewidth=3, markersize=7, label='$\Delta f$ (MD)')
plt.ylabel('$\omega_d$ (GHz)', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=18)
plt.xlabel('$\zeta_{ZZ} $ (MHz)', fontsize=18)
plt.xlim([-0.2e7, 0.2e7])
#plt.ylim([6.488, 6.523])
#plt.legend(fontsize=18)
plt.grid(True)

plt.tight_layout()
plt.show()

#%%
#%% 6. [결과 2] 특정 Base Freq에서 Phase Difference vs. Duration 플롯

# --- (PD) 데이터에서 가장 큰 차이를 보인 인덱스 선택 ---
try:
    # --- (MD) 데이터에서 가장 큰 차이를 보인 인덱스 선택 ---
    idx = np.nanargmin(np.abs(delta_f_mhz))
    idx = 3
    bf_example = bf[idx]

    A_g, f_g, phi_g, C_g = fits_g[idx]
    A_e, f_e, phi_e, C_e = fits_e[idx]
    
    phase_g_t = 2 * np.pi * f_g * t + phi_g
    phase_e_t = 2 * np.pi * f_e * t + phi_e
    delta_phase_t = np.unwrap(phase_g_t) - np.unwrap(phase_e_t)

    # --- 플로팅 ---
    fig, axes = plt.subplots(2, figsize=(14, 7))
    plt.suptitle("Example: Fit vs. Data and Phase Accumulation Difference", fontsize=16, y=1.02)

    # (3) MD - 데이터와 피팅 비교
    axes[0].plot(t, Z_g[idx], 'b.', alpha=0.5, label='g data')
    axes[0].plot(t, cos_func(t, A_g, f_g, phi_g, C_g), 'b-', label='g fit')
    axes[0].plot(t, Z_e[idx], 'r.', alpha=0.5, label='e data')
    axes[0].plot(t, cos_func(t, A_e, f_e, phi_e, C_e), 'r-', label='e fit')
    axes[0].set_title(f"MD Data vs. Fit (at bf={bf_example:.3f} GHz)")
    axes[0].set_xlabel('Duration (ns)')
    axes[0].set_ylabel('Population')
    axes[0].legend()

    # (4) MD - 위상 차이 플롯
    axes[1].plot(t, delta_phase_t, 'm-')
    axes[1].set_title(f"MD Phase Difference vs. Time (at bf={bf_example:.3f} GHz)")
    axes[1].set_xlabel('Duration (ns)')
    axes[1].set_ylabel('$\Delta \phi (t) = \phi_g(t) - \phi_e(t)$ (rad)')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

except ValueError as e:
    print(f"\n[Error] Plot 2 생성 실패: {e}")
    print("피팅이 실패하여 (NaN 반환) 최대값을 찾을 수 없습니다. 피팅 헬퍼 함수의 초기값을 확인하세요.")
# %%
