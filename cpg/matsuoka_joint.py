import math

class MatsuokaJoint:
    def __init__(self):
        self.beta = 2.5
        self.u0 = 1.0
        self.v1 = 1.0  # 1.0
        self.v2 = 0.0
        self.w21 = -2.0
        self.w12 = -2.0
        self.tu = 0.025
        self.tv = 0.3
        self.u1 = 0.0
        self.u2 = 1.0  # 1.0
        self.y1 = 0
        self.y2 = 0

    def get_output(self, input1, input2, timestep, tonic=1.0):
        self.u0 = tonic
        du1 = (1/self.tu)*(-self.u1-self.beta*self.v1+self.w12*self.y2+self.u0+input1)
        du2 = (1/self.tu)*(-self.u2-self.beta*self.v2+self.w21*self.y1+self.u0+input2)
        self.u1 = self.u1 + timestep*du1
        self.u2 = self.u2 + timestep*du2
        dv1 = 1/self.tv*(-self.v1+self.y1)
        dv2 = 1/self.tv*(-self.v2+self.y2)
        self.v1 = self.v1 + timestep*dv1
        self.v2 = self.v2 + timestep*dv2
        self.y1 = max(self.u1,0)
        self.y2 = max(self.u2,0)
        y = self.y2 - self.y1
        if math.isnan(y):
            y = 0
        elif(y>10**4):
            y = 10**4
        elif(y<-10**4):
            y=-10**4
        return y