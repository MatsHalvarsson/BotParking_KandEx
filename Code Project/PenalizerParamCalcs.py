from numpy import square, sqrt



class Penner:

    def __init__(self, e0, ang_max, dt, u_max=3.0, w_max=3.0, L_mid=1.0, L_upper_bound=1.5, cost_scaling=1.0, epsilon=1e-6):
        self.e0 = e0
        self.ang_max = ang_max
        self.dt = dt
        self.u_max = u_max
        self.w_max = w_max
        self.L_mid = L_mid
        self.L_upper_bound = L_upper_bound
        self.cost_scaling = cost_scaling
        self.epsilon = epsilon

        self.lu = L_mid / u_max
        self.lw = L_mid / w_max
        self.l_squares = [square(self.lu), square(self.lw)]
    
    @property
    def p1(self):
        """number of updates for traversing e0 distance with u_max"""
        return  self.e0 / (self.dt * self.u_max)
    
    @property
    def p2(self):
        """number of updates for rotating ang_max with w_max (rotating half a circle)"""
        return sqrt(2) * self.ang_max / (self.dt * self.w_max)
    
    def signal_Lcost_transgression(self, signal):
        return self.signal_L(signal) - self.L_upper_bound
    
    def penalty_on_Lupp_plus_Lmid(self):
        return self.penalty(self.L_mid + self.L_upper_bound)
    
    def signal_L(self, signal):
        return sqrt(sum(square(signal) * self.l_squares))
    
    def penalty(self, signal_L): #overload this in subclasses
        return self.cost_scaling * max(signal_L - self.L_upper_bound, 0.0)
    
    def __call__(self, old_state, action, new_state):
        return - self.penalty(self.signal_L(action))


if __name__ == "__main__":

    #constants defined elsewhere
    e0 = 1.0
    ang_max = 1.0

    #elsewhere constant tied to resolution
    dt = 0.01

    #hyper params
    u_max = 2.5
    w_max = 3.0
    L_mid = 1.0

    #hyper params 2
    L_upper_bound = 1.5
    L_lower_bound = 0.5
    #or
    interval = 0.5 # L_mid +- interval*L_mid

    L_sev = L_upper_bound + L_mid
    cost_diff_scaling = 2.0

    #Desired to have
    # L > L_upper_bound => penalty > 0
    # L > L_sev => penalty severe (reasonably)
    # -- and this could happen when ex L_upper_bound + L_mid
    # -- which would translate to (u_sev, w_sev) = (1 + L_upp/L_mid) * (u_Lmid, w_Lmid)

    #derived constants
    p1 = e0 / dt / u_max #number of updates for traversing e0 distance with u_max
    p2 = sqrt(2) * ang_max / dt / w_max #number of updates for rotating ang_max with w_max (rotating half a circle)
    lu = L_mid / u_max #relative weight on u in norm
    lw = lu * u_max / w_max #relative weight on w in norm
    # relation lu/lw is directly tied to p1/p2
    # -- how far are we allowed to get in the same time we can rotate half a circle
    # -- or, how much u do we have versus how much w when the cost L is equally attributed to both u and w

    #derived situations
    u_half = L_mid / lu / sqrt(2)
    w_half = L_mid / lw / sqrt(2)

    u_max_L_upp = L_upper_bound / lu
    u_half_L_upp = L_upper_bound / lu / sqrt(2)
    w_max_L_upp = L_upper_bound / lw
    w_half_L_upp = L_upper_bound / lw / sqrt(2)

    u_max_L_sev = L_sev / lu
    u_half_L_sev = L_sev / lu /sqrt(2)
    w_max_L_sev = L_sev / lw
    w_half_L_sev = L_sev / lw / sqrt(2)

    #on penalty size
    linear_cost_sev = L_sev - L_upper_bound
    quadratic_cost_sev = square(linear_cost_sev)
    scaled_linear_cost_sev = cost_diff_scaling * linear_cost_sev
    scaled_quadratic_cost_sev = square(scaled_linear_cost_sev)


    #print report:
    print("")
    print("p1: " + str(p1)[0:min(len(str(p1)), 4)] + ", p2: " + str(p2)[0:min(len(str(p2)), 4)])
    print("")
    print("L mid: (u_max, u_half, w_half, w_max) = (" + str(u_max)[0:min(len(str(u_max)), 4)] + ", " + \
          str(u_half)[0:min(len(str(u_half)), 4)] + ", " + str(w_half)[0:min(len(str(w_half)), 4)] + ", " + \
          str(w_max)[0:min(len(str(w_max)), 4)] + ")")
    print("L upper bound: (u_max, u_half, w_half, w_max) = (" + str(u_max_L_upp)[0:min(len(str(u_max_L_upp)), 4)] + ", " + \
          str(u_half_L_upp)[0:min(len(str(u_half_L_upp)), 4)] + ", " + str(w_half_L_upp)[0:min(len(str(w_half_L_upp)), 4)] + ", " + \
            str(w_max_L_upp)[0:min(len(str(w_max_L_upp)), 4)] + ")")
    print("L severe: (u_max, u_half, w_half, w_max) = (" + str(u_max_L_sev)[0:min(len(str(u_max_L_sev)), 4)] + ", " + \
          str(u_half_L_sev)[0:min(len(str(u_half_L_sev)), 4)] + ", " + str(w_half_L_sev)[0:min(len(str(w_half_L_sev)), 4)] + ", " + \
            str(w_max_L_sev)[0:min(len(str(w_max_L_sev)), 4)] + ")")
    print("")
    print("Linear cost L severe: " + str(linear_cost_sev)[0:min(len(str(linear_cost_sev)), 4)])
    print("Quadratic cost L severe: " + str(quadratic_cost_sev)[0:min(len(str(quadratic_cost_sev)), 4)])
    print("Scaled Linear cost L severe: " + str(scaled_linear_cost_sev)[0:min(len(str(scaled_linear_cost_sev)), 4)])
    print("Scaled Quadratic cost L severe: " + str(scaled_quadratic_cost_sev)[0:min(len(str(scaled_quadratic_cost_sev)), 4)])