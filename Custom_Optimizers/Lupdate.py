import numpy as np
import math

def L_update_linear(L, M):
    return L + M

def L_update_1_step(L, M, lr_mean): #Go up when lr mean goes down
    if len(lr_mean) >= 2:
        ratio = lr_mean[-1]/lr_mean[-2] #Ratio between current and previous lr mean
    else:
        ratio = 1
    delta_L = L*(1 - ratio) #Once rounded, gives the number of M to increase or decrease L with, negative when lr goes up
    if delta_L <= 0:
        return L + M
    else:
        return L - M

def L_update_lr_based(Lcont, M, lr_mean, s_plus, s_min, L_update_step): #Go up when lr mean goes down
    if len(lr_mean) >= 2:
        ratio = lr_mean[-1]/lr_mean[-2] #Ratio between current and previous lr mean
    else:
        ratio = 1
    delta_L = Lcont*(1-ratio) 
    if delta_L > 0: 
            Lcont = Lcont + delta_L*s_plus
    else:
        Lcont = Lcont + delta_L*s_min
    Lrest = Lcont % M     #Remainder of L that makes it undivisible by M
    Lint = Lcont - Lrest  #Highest value of L that is still divisible by M
    if L_update_step == 'round nearest integer':
        Lnew = Lint + round(Lrest/M) * M  #L rounded to nearest M  
        return Lnew, Lcont
    elif L_update_step == 'round up': #Matig nuttig nu waarschijnlijk
        if delta_L < 0:
            Lnew = Lint + math.ceil(Lrest/M)*M
        else:
            Lnew = Lint + math.floor(Lrest/M)*M
        return Lnew, Lcont
    else:
        print('ERROR: Invalid L update step. Try: round nearest integer, round up, 1 step.')

def L_update_set_per_epoch(epoch_schedule, L_schedule):
    pass

class LUPDATE():
    def __init__(self, M = 100, L = 200, max_L = 50000, min_L = 0, lr_mean = [], s_plus = 1, s_min = 's_plus', 
                 L_continues = 0,
                 L_update_type = 'linear', L_update_step = '1 step'):
        self.M = M
        self.L = L
        self.Lcont = L_continues
        self.max_L = max_L
        self.min_L = max(min_L,2*M)
        self.lr_mean = lr_mean
        self.s_plus = s_plus
        if s_min == 's_plus':
            self.s_min = s_plus
        elif s_min == 's_plus/2':
            self.s_min = s_plus/2
        elif s_min == 's_plus/10':
            self.s_min = s_plus/10
        elif s_min == 's_plus/100':
            self.s_min = s_plus/100
        else:
            self.s_min = s_min
        self.L_update_type = L_update_type
        self.L_update_step = L_update_step

    def L_update(self):
        old_L = self.L
        old_Lcont = self.Lcont
        #print(self.s_min, self.s_plus)
        if self.L_update_type == 'linear':
            self.L = L_update_linear(self.Lcont, self.M)
        elif self.L_update_type == 'lr based':
            if self.L_update_step == '1 step':
                self.L = L_update_1_step(self.L, self.M, self.lr_mean)
            else:
                self.L, self.Lcont = L_update_lr_based(self.Lcont, self.M, self.lr_mean, self.s_plus, self.s_min, self.L_update_step)
        else:
            print('ERROR: Invalid L update type. Try: linear, lr based.')
        if old_L != self.L:
            print('Previous L = ', old_L, ', New L = ', self.L)
            print('Previous L continues = ', old_Lcont, ', New L continues = ', self.Lcont)
        return max(min(self.L, self.max_L), self.min_L), max(min(self.Lcont, self.max_L), self.min_L)
