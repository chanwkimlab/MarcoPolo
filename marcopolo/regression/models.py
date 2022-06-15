import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

torch.set_default_dtype(torch.float64)

class Masked_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mask):
        #print('aaaa')
        output=input
        ctx.save_for_backward(input, mask)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, mask = ctx.saved_tensors
        grad_input = grad_mask = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mul(mask)

        return grad_input, grad_mask
    
class Masked(nn.Module):    
    def __init__(self, mask):    
        super(Masked, self).__init__()
        
        self.mask = nn.Parameter(torch.Tensor(mask)==1, requires_grad=False)    
        
        
    def forward(self, input):
        return Masked_Function.apply(input, self.mask)

    def extra_repr(self):
        return 'mask={}'.format(self.mask.shape)


class Poisson_logprob(nn.Module):
    def __init__(self):
        super(Poisson_logprob,self).__init__()
        
    def forward(self,rate,value):
        #rate=rate.clamp(min=1e-3)+(-1)/rate.clamp(max=-1e-5)
        
        return (rate.log() * value) - rate - (value + 1).lgamma()
    
poisson_logprob=Poisson_logprob()


# In[8]:


class Poisson_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Y, X, s, delta_log, beta, mask, to_return='LL'):

        with torch.no_grad():
            #mu=torch.exp((X.matmul(beta)+torch.log(s.view(-1, 1))).unsqueeze(dim=1).repeat(1,delta_log.shape[0],1)+torch.exp(delta_log))
            mu=torch.exp((X.matmul(beta)+torch.log(s.view(-1, 1))).unsqueeze(dim=1).repeat(1,delta_log.shape[0],1)+torch.exp(delta_log)*mask)
            Y_extend=Y.unsqueeze(dim=1).repeat(1,mu.shape[1],1)
            Y_logprob=poisson_logprob(rate=mu,value=Y_extend) # (N,C,G)
            Y_logprob_reduce=Y_logprob.sum(axis=2) # (N,C)
            
            Y_logprob_reduce_reduce=torch.logsumexp(Y_logprob_reduce,dim=1).view(-1,1) # (N,1)
            
            LL=torch.sum(Y_logprob_reduce_reduce) # (1)
            
            gamma=torch.exp(Y_logprob_reduce-Y_logprob_reduce_reduce)
            A=mu-Y.unsqueeze(dim=1)        
            
            #gradient
            grad_delta_log=(A*gamma.unsqueeze(dim=2)).sum(axis=0)
            grad_beta=(X.unsqueeze(dim=2)@gamma.unsqueeze(dim=1)@A).sum(axis=0)
        
            ctx.save_for_backward(grad_delta_log,grad_beta)
            
        if to_return=='LL':
            return LL
        elif to_return=='gamma':
            return gamma
        else:
            raise

    @staticmethod
    def backward(ctx, grad_output):
        
        grad_Y = grad_X = grad_s = grad_delta_log = grad_beta = grad_mask=None
        grad_delta_log,grad_beta = ctx.saved_tensors

        return grad_Y, grad_X, grad_s, grad_delta_log, grad_beta, grad_mask


class MarcoPoloModel(nn.Module):
    def __init__(self, Y, rho, X_col=5, delta_min=2):
        # Y,rho are needed for model parameter initialization
        super(MarcoPoloModel, self).__init__()

        # rho
        self.masked = Masked(rho)
        self.init_paramter_rho(rho)
        # delta
        self.delta_log_min = np.log(delta_min)  #
        self.delta_log = nn.Parameter(torch.Tensor(np.ones(rho.shape)), requires_grad=True)  # (C,G)
        self.init_parameter_delta_min(delta_min)
        #beta
        self.beta=nn.Parameter(torch.Tensor(np.ones((X_col,Y.shape[1]))),requires_grad=True) # (P,G)
        self.init_paramter_Y(Y)
        
    def init_paramter_rho(self,rho):
        self.masked.mask.data=torch.Tensor((rho==1)).to(self.masked.mask.device)
        
    def init_parameter_delta_min(self,delta_min):
        self.delta_log_min=np.log(delta_min) #
        if delta_min==0:
            #self.delta_log.data[:]=torch.Tensor(np.random.uniform(-2,2,size=self.delta_log.data.shape))
            self.delta_log.data=torch.Tensor(np.random.uniform(np.log(2)-0.1,np.log(2)+0.1,size=self.delta_log.shape)).to(self.delta_log.device) # (C,G)
        else:
            self.delta_log.data=torch.Tensor(np.random.uniform(self.delta_log_min-0.1,self.delta_log_min+0.1,size=self.delta_log.shape)).to(self.delta_log.device) # (C,G)
        self.delta_log.data=self.delta_log.data.clamp(min=self.delta_log_min)    

    def init_paramter_Y(self,Y):
        Y_colmean=np.mean(Y,axis=0)
        beta_init=np.hstack([((Y_colmean-Y_colmean.mean())/(np.std(Y_colmean) if len(Y_colmean)>1 else 1)).reshape(-1,1),                     np.zeros((Y.shape[1],self.beta.shape[0]-1))]).T      
        self.beta.data[:]=torch.Tensor(beta_init).to(self.beta.device)       
        
    def forward(self,Y,X,s, to_return='LL'):
        if to_return=='LL':
            delta_log_masked=self.masked(self.delta_log) #(C,G)
            #delta=torch.exp(delta_log_masked)*self.masked.mask
            LL=Poisson_Function.apply(Y, X, s, delta_log_masked, self.beta, self.masked.mask)
            return LL                      
        elif to_return=='gamma':
            with torch.no_grad():
                gamma=Poisson_Function.apply(Y, X, s, self.delta_log, self.beta, self.masked.mask, 'gamma')
            return gamma


if __name__ == '__main__':
    model = MarcoPoloModel(Y=np.ones((5, 5)), rho=np.ones((5, 5)))
    a = model(Y=torch.Tensor(np.ones((5, 5))), X=torch.Tensor(np.ones((5, 5))), s=torch.Tensor(np.ones((5, 1))))
    a.backward()

