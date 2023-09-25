import torch as th
import numpy as np

LOG_PDF_MAX = 4
LOG_PDF_MIN = -20

class KernelDensityEstimation1D():
    '''
        1-D KDE with Gaussian Kernel
    '''
    def __init__(self,delta:float=0.01) -> None:
        self.delta = delta # the bandwitch

    def pdf(self,u:th.Tensor, u_data:th.Tensor)->th.Tensor:
        """
            u.shape = (len(u),) , u_data.shape = (len(u_data), 1)
        """
        num = u_data.shape[1]
        u = u.unsqueeze(dim=1)
        return (th.exp(-((u-u_data)/self.delta)**2/2)).sum(dim=1).flatten()\
                /num/self.delta/np.sqrt(2*np.pi)
    
    def log_pdf(self,u:th.Tensor, u_data:th.Tensor)->th.Tensor:
        pdf = self.pdf(u=u,u_data=u_data)
        log_pdf  = pdf.log()
        log_pdf = th.clamp(log_pdf, LOG_PDF_MIN, LOG_PDF_MAX)
        return log_pdf, pdf

    def plot(self,u_min,u_max,u_data:th.Tensor) ->None:
        import matplotlib.pyplot as plt
        u_data = u_data.clone().detach().to("cpu")
        u = th.linspace(u_min,u_max,1000)
        log_pdf, pdf = self.log_pdf(u,u_data)
        fig, axes = plt.subplots(2,1,figsize=(9,6))
        
        axes[0].plot(u,pdf)
        axes[1].plot(u,log_pdf)
        plt.xlim(u_min,u_max)
        plt.show()
        plt.close()
        return

class KernelDensityEstimation():
    '''
        n-D KDE with Gaussian Kernel n>=2
    '''
    def __init__(self,u_dim:int, delta:float=0.01) -> None:
        self.delta = delta # the bandwitch
        self.u_dim = u_dim

    def pdf(self,u:th.Tensor, u_data:th.Tensor)->th.Tensor:
        """
            u.shape = (len(u), 1 ,u_dim) , u_data.shape = (len(u_data), sim_num , u_dim)
            len(u) = len(u_data) = batch_size
        """
        assert u.shape[2]==self.u_dim, "the u_dim should be {}, but get {}".format(u.shape[1], self.u_dim)
        sim_num = u_data.shape[1]
        # (batch_size, sim_num, u_dim)
        pdf = ((-((u-u_data)/self.delta)**2/2).sum(dim=2).exp()).sum(dim=1)\
                /(sim_num*self.delta**self.u_dim)/(np.sqrt(2*np.pi)**self.u_dim)     
        return pdf
    
    def log_pdf(self,u:th.Tensor, u_data:th.Tensor)->th.Tensor:
        """
            u.shape = (len(u), 1 ,u_dim) , u_data.shape = (len(u), sim_num , u_dim)
            len(u) = batch_size
        """
        log_pdf = self.pdf(u=u,u_data=u_data).log()
        log_pdf = th.clamp(log_pdf, LOG_PDF_MIN*self.u_dim, LOG_PDF_MAX*self.u_dim)
        return log_pdf


# testing code
if __name__ == '__main__':  
    kde = KernelDensityEstimation1D()
    u_data = (th.randn((1,100000))*5).tanh()
    #u_data = th.randn(1,100000)
    u_pdf = kde.plot(-2,2,u_data)

    
    # u_data = (th.randn((5,100000,2)))
    # kde = KernelDensityEstimation(u_dim= u_data.shape[2],delta=0.1)
    # u_pdf = kde.log_pdf(u=th.zeros([5,1,2]),u_data=u_data)
    # print(u_pdf)
    # print(np.log(1/2/np.pi))
    
