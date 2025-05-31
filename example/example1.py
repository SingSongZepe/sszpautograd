
import sys  
import os  

# Add the parent directory to sys.path where contains the tensor.py  
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  

import numpy as np
import matplotlib.pyplot as plt
from tensor import *

# this file analyze the properties of beta_1 to its variables
def beta_1_func(c_1a: float, rho_ct: float, u_cp: float, L_cp: float):
    return math.atan(c_1a/(rho_ct * u_cp + L_cp/2/u_cp))

def analyze_dbeta_1_dc_1a():

    rho_ct_val = 0.58
    u_cp_val = 409
    L_ct_val = 39562

    rho_ct = Tensor(rho_ct_val)
    u_cp = Tensor(u_cp_val)
    L_ct = Tensor(L_ct_val)

    c_1a_vals = np.linspace(200, 250, 50)
    dbeta_1_dc_1a_list = []
    for c_1a_val in c_1a_vals:
        c_1a = Tensor(c_1a_val)
        
        beta_1 = Tensor.atan(c_1a / (rho_ct * u_cp + L_ct / Tensor(2) / u_cp))

        beta_1.backward()

        log.ln(c_1a.grad)

        dbeta_1_dc_1a_list.append(c_1a.grad)

    plt.plot(c_1a_vals, dbeta_1_dc_1a_list)
    plt.xlabel('c_1a m/s')
    plt.ylabel('dbeta_1/dc_1a')

    beta_1_vals = [beta_1_func(c_1a_val, rho_ct_val, u_cp_val, L_cp_val) for c_1a in c_1a_vals]
    # plt.plot(c_1a_vals, beta_1_vals)

    plt.legend()
    plt.title('deriviate of beta_1 to c_1a')

    plt.show()

def analyze_dbeta_1_drho_ct():

    c_1a_val = 220
    u_cp_val = 409
    L_ct_val = 39562

    c_1a = Tensor(c_1a_val)
    u_cp = Tensor(u_cp_val)
    L_ct = Tensor(L_ct_val)

    rho_ct_vals = np.linspace(0.5, 0.7, 50)
    dbeta_1_drho_ct_list = []
    for rho_ct_val in rho_ct_vals:
        rho_ct = Tensor(rho_ct_val)

        beta_1 = Tensor.atan(c_1a / (rho_ct * u_cp + L_ct / Tensor(2) / u_cp))

        beta_1.backward()

        dbeta_1_drho_ct_list.append(rho_ct.grad)
    
    plt.plot(rho_ct_vals, dbeta_1_drho_ct_list)
    plt.xlabel('rho_ct')
    plt.ylabel('dbeta_1/drho_ct')

    beta_1_vals = [beta_1_func(c_1a_val, rho_ct_val, u_cp_val, L_ct_val) for rho_ct_val in rho_ct_vals]
    # plt.plot(c_1a_vals, beta_1_vals)

    plt.legend()
    plt.title('deriviate of beta_1 to rho_ct')

    plt.show()

def main():
    # analyze_dbeta_1_dc_1a()
    analyze_dbeta_1_drho_ct()

if __name__ == '__main__':
    main()