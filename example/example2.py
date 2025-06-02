
import sys  
import os  

# Add the parent directory to sys.path where contains the tensor.py  
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  

import numpy as np
import matplotlib.pyplot as plt
from tensor import *

DEBUG_ITEARTION_VALUE_INFO = True

def clamp(value, a, b):  
    '''
    Clamp a value to the range [a, b].
    '''  
    return max(a, min(value, b))  

def analyze_pitot_ct():
    # define constant parameters
    D_1cp = 0.711
    D_2cp = 0.708
    
    k = 1.4
    R = 287

    n = 11000
    
    Ttot_1_val = 391.66
    ptot_1_val = 260111    

    u_1cp_val = math.pi * D_1cp * n / 60
    u_2cp_val = math.pi * D_2cp * n / 60
    u_cp_val = (u_1cp_val + u_2cp_val) / 2

    # define input parameters
    rho_ct_val = 0.58
    L_ct_val = 39562
    c_1a_val = 220
    c_2a_val = 214.17

    rho_ct_val = 0.58
    L_ct_val = 39562
    c_1a_val = 220
    c_2a_val = 214.17

    # gradient descent optimization
    # learning rate respectively to
    # rho_ct, L_ct, c_1a, c_2a
    [lr1, lr2, lr3, lr4] = (0.1, 100, 1, 1)

    [rho_ct_val_max, rho_ct_val_min] = (0.75, 0.55)
    [L_ct_val_max, L_ct_val_min] = (43562, 33562)
    [c_1a_val_max, c_1a_val_min] = (230, 200)
    [c_2a_val_max, c_2a_val_min] = (225, 195)

    # the minimum required difference c_1a - c_2a >= 4
    DIFFERENCE_c_1a_c_2a = 4

    x_collector = list(range(1, 101))
    pitot_ct_collector = []

    ITERATION_PER_COLLECTION = 200

    total_iteration_time = ITERATION_PER_COLLECTION * len(x_collector)
    log.ln(f'total iteration time: {total_iteration_time}')
    
    for i in range(total_iteration_time):
        log.ln(f'iteration step: {i}')

        # c_2a_val = c_1a_val - DIFFERENCE_c_1a_c_2a

        # define input tensors
        rho_ct = Tensor(rho_ct_val)
        L_ct = Tensor(L_ct_val)
        c_1a = Tensor(c_1a_val)
        c_2a = Tensor(c_2a_val)

        # build the result tensor by some intermediate tensors

        c_1u = (Tensor(1)-rho_ct-L_ct/Tensor(2)/Tensor(u_cp_val)**2)*Tensor(u_cp_val)
        c_2u = (L_ct + c_1u * Tensor(u_1cp_val)) / Tensor(u_2cp_val)

        c_1 = Tensor.sqrt(c_1u ** 2 + c_1a ** 2)
        c_2 = Tensor.sqrt(c_2u ** 2 + c_2a ** 2)

        w_1u = Tensor(u_1cp_val) - c_1u
        w_1 = Tensor.sqrt(c_1a ** 2 + w_1u ** 2)

        T_1 = Tensor(Ttot_1_val) - c_1 ** 2 / Tensor(2*k*R/(k-1))
        Ttot_1w = T_1 + w_1 ** 2 / Tensor(2*k*R/(k-1))

        lambda_w1 = w_1 / Tensor.sqrt(Tensor(2*k/(k+1)*R) * Ttot_1w)
        pi_lambda_w1 = (Tensor(1) - Tensor((k-1)/(k+1)) * lambda_w1 ** 2) ** (k/(k-1))

        sigma_HA = Tensor(-0.0229)*lambda_w1**3 + Tensor(0.0246)*lambda_w1**2 - Tensor(0.0396)*lambda_w1 + Tensor(1.0116)

        lambda_c1 = c_1 / Tensor.sqrt(Tensor(2*k/(k+1)*R) * Tensor(Ttot_1_val))
        pi_lambda_c1 = (Tensor(1) - Tensor((k-1)/(k+1)) * lambda_c1 ** 2) ** (k/(k-1))
        
        p_1 = Tensor(ptot_1_val) * pi_lambda_c1
        ptot_1w = p_1 / pi_lambda_w1

        rho_1 = p_1 / Tensor(R) / T_1

        w_2u = Tensor(u_2cp_val) - c_2u
        w_2 = Tensor.sqrt(c_1a ** 2 + w_2u ** 2)

        Ttot_2w = Ttot_1w - Tensor(u_1cp_val ** 2 - u_2cp_val ** 2) / Tensor(2*k*R/(k-1))
        T_2 = Ttot_2w - w_2 ** 2 / Tensor(2*k*R/(k-1))
        Ttot_2 = T_2 + c_2 ** 2 / Tensor(2*k*R/(k-1))

        lambda_w2 = w_2 / Tensor.sqrt(Tensor(2*k/(k+1)*R) * Ttot_2w)
        pi_lambda_w2 = (Tensor(1) - Tensor((k-1)/(k+1)) * lambda_w2 ** 2) ** (k/(k-1))

        lambda_c2 = c_2 / Tensor.sqrt(Tensor(2*k/(k+1)*R) * Ttot_2)
        pi_lambda_c2 = (Tensor(1) - Tensor((k-1)/(k+1)) * lambda_c2 ** 2) ** (k/(k-1))
        
        sigma_PK = Tensor(-0.0229)*lambda_c2**3 + Tensor(0.0246)*lambda_c2**2 - Tensor(0.0396)*lambda_c2 + Tensor(1.0116)

        ptot_2w = (ptot_1w + rho_1 * Tensor((u_2cp_val ** 2 - u_1cp_val ** 2)/2)) * sigma_PK

        p_2 = ptot_2w * pi_lambda_w2
        ptot_2 = p_2 / pi_lambda_c2

        ptot_3 = ptot_2 * sigma_HA

        pitot_ct = ptot_3 / Tensor(ptot_1_val)

        pitot_ct.backward()

        # use gradient descent to optimize the value of pitot_ct
        rho_ct_val = clamp(rho_ct_val + lr1 * rho_ct.grad, rho_ct_val_min, rho_ct_val_max)
        L_ct_val   = clamp(L_ct_val + lr2 * L_ct.grad, L_ct_val_min, L_ct_val_max)
        c_1a_val   = clamp(c_1a_val + lr3 * c_1a.grad, c_1a_val_min, c_1a_val_max)
        c_2a_val   = clamp(c_2a_val + lr4 * c_2a.grad, c_2a_val_min, c_2a_val_max)

        # under some circumstance, we want c_2a_val < c_1a_val by some difference
        # if c_1a_val - c_2a_val < DIFFERENCE_c_1a_c_2a:
        #     mid = (c_1a_val + c_2a_val) / 2
        #     c_1a_val = mid + DIFFERENCE_c_1a_c_2a / 2 
        #     c_2a_val = mid - DIFFERENCE_c_1a_c_2a / 2
        
        if DEBUG_ITEARTION_VALUE_INFO:
            log.ln(f'current dpitot_ct/drho_ct: {rho_ct.grad}')
            log.ln(f'current dpitot_ct/dL_ct: {L_ct.grad}')
            log.ln(f'current dpitot_ct/dc_1a: {c_1a.grad}')
            log.ln(f'current dpitot_ct/dc_2a: {c_2a.grad}')

            log.ln(f'current rho_ct value: {rho_ct_val}')
            log.ln(f'current L_ct value: {L_ct_val}')
            log.ln(f'current c_1a value: {c_1a_val}')
            log.ln(f'current c_2a value: {c_2a_val}')
            log.ln(f'pitot_ct value: {pitot_ct.val}')

            log.ln()

        if i % ITERATION_PER_COLLECTION == 0:
            pitot_ct_collector.append(pitot_ct.val)

    plt.plot(x_collector, pitot_ct_collector, marker='x')

    plt.xlabel('iteration /time')
    plt.ylabel('pitot_ct')

    plt.show()


def main():
    analyze_pitot_ct()

if __name__ == '__main__':
    main()

