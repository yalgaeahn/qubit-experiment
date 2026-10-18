_kps_freq=_kps_np.linspace(4.67628e9,4.68428e9,161)
_kps_baseline=_kps_run_scan('01_drive_off',_kps_freq,None,count=512)
_kps_low=_kps_run_scan('02_minus49p46',_kps_freq,-49.46,count=1024)


_kps_fine_freq=_kps_np.linspace(4.6793e9,4.6809e9,81)
_kps_fine=_kps_run_scan('03_fine_minus49p46',_kps_fine_freq,-49.46,count=2048)
_kps_power_check=_kps_run_scan('04_reverse_minus43p46',_kps_fine_freq[::-1],-43.46,count=1024)


_kps_final=_kps_run_scan('05_confirm_low_reverse',_kps_fine_freq[::-1],-49.46,count=2048)
